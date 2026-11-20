"""The Swagger upload form and the folder-upload helper."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.core.config import Settings
from app.main import create_app
from scripts import upload_folder


def test_swagger_shows_a_file_picker_for_uploads() -> None:
    spec = create_app(Settings(_env_file=None)).openapi()  # type: ignore[call-arg]
    body = spec["paths"]["/api/v1/papers/upload"]["post"]["requestBody"]["content"]
    ref = body["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    props = spec["components"]["schemas"][ref]["properties"]
    # Swagger UI renders a "Choose File" button only for format: binary.
    assert props["files"]["items"] == {"type": "string", "format": "binary"}
    assert props["collection"]["examples"] == ["rag-papers"]


def test_find_pdfs_and_batches(tmp_path: Path) -> None:
    for name in ("b.pdf", "a.PDF", "notes.txt"):
        (tmp_path / name).write_bytes(b"%PDF-1.7")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.pdf").write_bytes(b"%PDF-1.7")
    assert [p.name for p in upload_folder.find_pdfs(tmp_path)] == ["a.PDF", "b.pdf"]
    assert len(upload_folder.find_pdfs(tmp_path, recursive=True)) == 3
    assert [len(b) for b in upload_folder.batches(list(range(23)), 10)] == [10, 10, 3]  # type: ignore[arg-type]


def test_upload_batch_handles_accepted_and_all_duplicate(tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    def handler(request: httpx.Request) -> httpx.Response:
        assert b'filename="p.pdf"' in request.content and b"rag-papers" in request.content
        if handler.calls == 0:  # type: ignore[attr-defined]
            handler.calls += 1  # type: ignore[attr-defined]
            return httpx.Response(
                202,
                json={
                    "accepted": 1,
                    "results": [{"filename": "p.pdf", "status": "accepted", "paper_id": "x"}],
                },
            )
        return httpx.Response(
            409,
            content=json.dumps(
                {
                    "error": {
                        "details": [
                            {
                                "filename": "p.pdf",
                                "error_code": "duplicate",
                                "paper_id": "x",
                                "message": "already uploaded",
                            }
                        ]
                    }
                }
            ),
        )

    handler.calls = 0  # type: ignore[attr-defined]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = upload_folder.upload_batch(client, "http://api", [pdf], "rag-papers")
        second = upload_folder.upload_batch(client, "http://api", [pdf], "rag-papers")
    assert first[0]["status"] == "accepted"
    assert second[0]["status"] == "duplicate" and second[0]["paper_id"] == "x"
