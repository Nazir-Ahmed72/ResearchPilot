"""Paper endpoints end-to-end through FastAPI (SQLite, in-memory Qdrant, fake embedder)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.resources import Resources
from app.workers.dispatch import InlineDispatcher
from tests.fakes import RecordingDispatcher
from tests.pdf_factory import make_paper_pdf

UPLOAD = "/api/v1/papers/upload"


def pdf_file(data: bytes | None = None, name: str = "paper.pdf", ctype: str = "application/pdf"):
    return ("files", (name, data if data is not None else make_paper_pdf(), ctype))


def test_upload_accepts_and_enqueues(client: TestClient, dispatcher: RecordingDispatcher) -> None:
    resp = client.post(UPLOAD, files=[pdf_file()], data={"collection": "rag"})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["accepted"] == 1
    result = body["results"][0]
    assert result["status"] == "accepted" and result["paper_id"] and result["job_id"]
    assert [str(p) for p, _ in dispatcher.calls] == [result["paper_id"]]

    detail = client.get(f"/api/v1/papers/{result['paper_id']}").json()
    assert detail["status"] == "pending"
    assert detail["collection"] == "rag"
    assert detail["page_count"] == 3
    assert detail["ingestion_jobs"][0]["task_id"] == "task-1"


def test_multiple_files_with_mixed_outcomes(client: TestClient) -> None:
    good = make_paper_pdf()
    resp = client.post(
        UPLOAD,
        files=[pdf_file(good, "a.pdf"), pdf_file(good, "copy.pdf"), pdf_file(b"nope", "b.pdf")],
    )
    assert resp.status_code == 202
    statuses = [r["status"] for r in resp.json()["results"]]
    assert statuses == ["accepted", "duplicate", "rejected"]
    results = resp.json()["results"]
    assert results[1]["paper_id"] == results[0]["paper_id"]
    assert results[2]["error_code"] == "invalid_pdf"


def test_duplicate_upload_returns_409_with_existing_id(client: TestClient) -> None:
    data = make_paper_pdf()  # generated PDFs embed a random ID, so reuse the exact bytes
    first = client.post(UPLOAD, files=[pdf_file(data)]).json()["results"][0]["paper_id"]
    resp = client.post(UPLOAD, files=[pdf_file(data, "renamed.pdf")])
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert err["code"] == "upload_rejected"
    assert err["details"][0]["paper_id"] == first


@pytest.mark.parametrize(
    ("file", "status"),
    [
        (("files", ("notes.txt", b"hello", "text/plain")), 415),
        (("files", ("x.pdf", b"not a pdf", "application/pdf")), 400),
        (("files", ("big.pdf", b"%PDF-" + b"0" * (3 * 1024 * 1024), "application/pdf")), 413),
    ],
)
def test_rejections_map_to_status_codes(client: TestClient, file, status: int) -> None:
    resp = client.post(UPLOAD, files=[file])
    assert resp.status_code == status
    assert resp.json()["error"]["code"] == "upload_rejected"


def test_too_many_files_rejected(client: TestClient) -> None:
    resp = client.post(UPLOAD, files=[pdf_file(name=f"{i}.pdf") for i in range(4)])
    assert resp.status_code == 400
    assert "At most 3" in resp.json()["error"]["message"]


def test_no_files_is_validation_error(client: TestClient) -> None:
    assert client.post(UPLOAD).status_code == 422


def test_broker_down_marks_paper_failed(resources: Resources) -> None:
    resources._dispatcher = RecordingDispatcher(fail=True)
    with TestClient(FastAPIApp(resources)) as client:
        resp = client.post(UPLOAD, files=[pdf_file()])
        assert resp.status_code == 503
        (result,) = resp.json()["error"]["details"]
        assert result["error_code"] == "dependency_unavailable"
        assert "reingest" in result["message"]
        detail = client.get(f"/api/v1/papers/{result['paper_id']}").json()
        assert detail["status"] == "failed"
        assert detail["ingestion_jobs"][0]["status"] == "failed"


def FastAPIApp(resources: Resources) -> FastAPI:  # noqa: N802 - reads like a constructor
    from app.main import create_app

    return create_app(resources=resources)


def test_list_and_filter(client: TestClient) -> None:
    client.post(UPLOAD, files=[pdf_file()], data={"collection": "a"})
    client.post(
        UPLOAD,
        files=[pdf_file(make_paper_pdf(title="Second Paper Title"))],
        data={"collection": "b"},
    )
    everything = client.get("/api/v1/papers").json()
    assert everything["total"] == 2
    only_a = client.get("/api/v1/papers", params={"collection": "a"}).json()
    assert only_a["total"] == 1 and only_a["items"][0]["collection"] == "a"
    assert client.get("/api/v1/papers", params={"collection": "zzz"}).json()["total"] == 0
    assert client.get("/api/v1/papers", params={"status": "ready"}).json()["total"] == 0
    paged = client.get("/api/v1/papers", params={"limit": 1, "offset": 1}).json()
    assert len(paged["items"]) == 1 and paged["total"] == 2
    assert client.get("/api/v1/papers", params={"status": "bogus"}).status_code == 422


def test_get_unknown_and_malformed_ids(client: TestClient) -> None:
    assert client.get("/api/v1/papers/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/api/v1/papers/not-a-uuid").status_code == 422


@pytest.fixture
def inline_client(resources: Resources) -> Iterator[TestClient]:
    """Real end-to-end: uploads are ingested in a background thread of the API process."""
    resources._dispatcher = InlineDispatcher(resources.ingestion_pipeline)
    with TestClient(FastAPIApp(resources)) as client:
        yield client


def _wait(resources: Resources) -> None:
    dispatcher = resources.dispatcher
    assert isinstance(dispatcher, InlineDispatcher)
    for future in dispatcher.futures.values():
        future.result(timeout=30)


def test_full_lifecycle_upload_ingest_reingest_delete(
    inline_client: TestClient, resources: Resources
) -> None:
    pid = inline_client.post(UPLOAD, files=[pdf_file()]).json()["results"][0]["paper_id"]
    _wait(resources)
    paper = inline_client.get(f"/api/v1/papers/{pid}").json()
    assert paper["status"] == "ready"
    assert paper["chunk_count"] > 0
    assert paper["title"] == "Hybrid Retrieval for Scientific Question Answering"
    assert resources.chunk_store.count(pid) == paper["chunk_count"]

    resp = inline_client.post(f"/api/v1/papers/{pid}/reingest")
    assert resp.status_code == 202
    _wait(resources)
    paper = inline_client.get(f"/api/v1/papers/{pid}").json()
    assert paper["status"] == "ready" and len(paper["ingestion_jobs"]) == 2
    assert resources.chunk_store.count(pid) == paper["chunk_count"]

    assert inline_client.delete(f"/api/v1/papers/{pid}").status_code == 204
    assert inline_client.get(f"/api/v1/papers/{pid}").status_code == 404
    assert resources.chunk_store.count(pid) == 0
    assert list(resources.settings.upload_dir.iterdir()) == []
    assert inline_client.delete(f"/api/v1/papers/{pid}").status_code == 404


def test_delete_while_processing_conflicts(client: TestClient, resources: Resources) -> None:
    from app.database.models import Paper, PaperStatus

    pid = client.post(UPLOAD, files=[pdf_file()]).json()["results"][0]["paper_id"]
    with resources.session_factory() as s:
        s.get(Paper, __import__("uuid").UUID(pid)).status = PaperStatus.PROCESSING
        s.commit()
    resp = client.delete(f"/api/v1/papers/{pid}")
    assert resp.status_code == 409
    assert client.post(f"/api/v1/papers/{pid}/reingest").status_code == 409


def test_failed_vector_delete_keeps_paper_retryable(
    client: TestClient, resources: Resources, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = client.post(UPLOAD, files=[pdf_file()]).json()["results"][0]["paper_id"]

    def broken(_: object) -> None:
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(resources.chunk_store, "delete_paper", broken)
    resp = client.delete(f"/api/v1/papers/{pid}")
    assert resp.status_code == 503
    assert client.get(f"/api/v1/papers/{pid}").json()["status"] == "deleting"

    monkeypatch.undo()
    assert client.delete(f"/api/v1/papers/{pid}").status_code == 204
