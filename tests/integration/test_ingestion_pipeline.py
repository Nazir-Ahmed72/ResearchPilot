"""Ingestion pipeline against SQLite + in-memory Qdrant + fake embedder."""

from __future__ import annotations

import hashlib
import uuid

import pytest
from qdrant_client import models

from app.core.resources import Resources
from app.database.models import IngestionJob, JobStatus, Paper, PaperStatus
from app.ingestion.errors import PermanentIngestionError, TransientIngestionError
from app.vectorstore.qdrant_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantChunkStore,
    VectorStoreConfigError,
)
from tests.pdf_factory import make_blank_pdf, make_paper_pdf


def _create_paper(resources: Resources, data: bytes) -> tuple[uuid.UUID, uuid.UUID]:
    pid, jid = uuid.uuid4(), uuid.uuid4()
    key = resources.storage.save(pid, data)
    with resources.session_factory() as s:
        paper = Paper(
            id=pid,
            content_hash=hashlib.sha256(data).hexdigest(),
            original_filename="p.pdf",
            storage_path=key,
            file_size_bytes=len(data),
        )
        s.add_all([paper, IngestionJob(id=jid, paper=paper)])
        s.commit()
    return pid, jid


def _state(resources: Resources, pid: uuid.UUID, jid: uuid.UUID) -> tuple[Paper, IngestionJob]:
    with resources.session_factory() as s:
        return s.get(Paper, pid), s.get(IngestionJob, jid)


def _points(resources: Resources, pid: uuid.UUID) -> list[models.Record]:
    store = resources.chunk_store
    records, _ = resources.qdrant.scroll(
        store.collection,
        scroll_filter=store.paper_filter(pid),
        limit=1000,
        with_payload=True,
        with_vectors=True,
    )
    return records


def test_successful_ingestion_indexes_chunks_with_provenance(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_paper_pdf())
    result = resources.ingestion_pipeline.run(pid, jid)

    paper, job = _state(resources, pid, jid)
    assert result is not None and result.chunk_count == paper.chunk_count > 0
    assert paper.status is PaperStatus.READY and job.status is JobStatus.SUCCEEDED
    assert paper.title == "Hybrid Retrieval for Scientific Question Answering"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert (paper.publication_year, paper.doi, paper.page_count) == (
        2023,
        "10.1234/rp.2023.0042",
        3,
    )
    assert job.attempts == 1 and job.started_at and job.finished_at
    assert set(job.stats["timings_ms"]) == {"extract", "chunk", "embed", "index"}

    points = _points(resources, pid)
    assert len(points) == paper.chunk_count
    payload = points[0].payload
    for key in (
        "chunk_id",
        "paper_id",
        "page_start",
        "page_end",
        "section",
        "title",
        "text",
        "authors",
        "year",
        "doi",
        "source",
    ):
        assert key in payload
    assert payload["paper_id"] == str(pid)
    assert all(p.id == p.payload["chunk_id"] for p in points)
    assert all(DENSE_VECTOR in p.vector and SPARSE_VECTOR in p.vector for p in points)
    limitations = [p for p in points if p.payload["section"] == "4 Limitations"]
    assert limitations and limitations[0].payload["page_start"] == 3
    assert not any(p.payload["section"] == "References" for p in points)


def test_reingestion_is_idempotent(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_paper_pdf())
    resources.ingestion_pipeline.run(pid, jid)
    first_ids = sorted(str(p.id) for p in _points(resources, pid))
    resources.ingestion_pipeline.run(pid, jid)
    second_ids = sorted(str(p.id) for p in _points(resources, pid))
    assert first_ids == second_ids
    assert resources.chunk_store.count() == len(first_ids)
    _, job = _state(resources, pid, jid)
    assert job.attempts == 2


def test_two_papers_are_isolated(resources: Resources) -> None:
    a = _create_paper(resources, make_paper_pdf())
    b = _create_paper(resources, make_paper_pdf(title="A Different Paper About Graphs"))
    resources.ingestion_pipeline.run(*a)
    resources.ingestion_pipeline.run(*b)
    resources.chunk_store.delete_paper(a[0])
    assert resources.chunk_store.count(a[0]) == 0
    assert resources.chunk_store.count(b[0]) > 0


def test_pdf_without_text_fails_permanently(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_blank_pdf(2))
    with pytest.raises(PermanentIngestionError):
        resources.ingestion_pipeline.run(pid, jid)
    paper, job = _state(resources, pid, jid)
    assert paper.status is PaperStatus.FAILED and job.status is JobStatus.FAILED
    assert "No extractable text" in paper.error_message
    assert resources.chunk_store.count(pid) == 0


class _ExplodingStore(QdrantChunkStore):
    def upsert(self, points, batch_size=64):  # type: ignore[no-untyped-def]
        list(points)
        raise ConnectionError("qdrant went away; api_key=supersecret")


def test_transient_failure_compensates_and_allows_retry(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_paper_pdf())
    pipeline = resources.ingestion_pipeline
    real_store = pipeline.store
    pipeline.store = _ExplodingStore(resources.qdrant, real_store.collection)

    with pytest.raises(TransientIngestionError):
        pipeline.run(pid, jid, final_attempt=False)
    paper, job = _state(resources, pid, jid)
    assert paper.status is PaperStatus.PENDING and job.status is JobStatus.QUEUED
    assert "supersecret" not in job.error_message  # secrets redacted in stored errors

    pipeline.store = real_store  # "Qdrant is back": the retry succeeds
    assert pipeline.run(pid, jid) is not None
    paper, job = _state(resources, pid, jid)
    assert paper.status is PaperStatus.READY and job.attempts == 2


def test_final_transient_failure_marks_failed(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_paper_pdf())
    resources.ingestion_pipeline.store = _ExplodingStore(resources.qdrant, "c")
    with pytest.raises(TransientIngestionError):
        resources.ingestion_pipeline.run(pid, jid, final_attempt=True)
    paper, job = _state(resources, pid, jid)
    assert paper.status is PaperStatus.FAILED and job.status is JobStatus.FAILED


def test_paper_deleted_before_start_is_skipped(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_paper_pdf())
    with resources.session_factory() as s:
        s.get(Paper, pid).status = PaperStatus.DELETING
        s.commit()
    assert resources.ingestion_pipeline.run(pid, jid) is None
    assert resources.chunk_store.count(pid) == 0


def test_paper_deleted_during_ingestion_leaves_no_vectors(resources: Resources) -> None:
    pid, jid = _create_paper(resources, make_paper_pdf())
    pipeline = resources.ingestion_pipeline
    original_upsert = pipeline.store.upsert

    def upsert_then_user_deletes(points, batch_size=64):  # type: ignore[no-untyped-def]
        n = original_upsert(points, batch_size)
        with resources.session_factory() as s:
            s.get(Paper, pid).status = PaperStatus.DELETING
            s.commit()
        return n

    pipeline.store.upsert = upsert_then_user_deletes  # type: ignore[method-assign]
    assert pipeline.run(pid, jid) is None
    assert resources.chunk_store.count(pid) == 0


def test_dimension_mismatch_is_reported(resources: Resources) -> None:
    store = resources.chunk_store
    store.ensure_collection(32)
    other = QdrantChunkStore(resources.qdrant, store.collection)
    with pytest.raises(VectorStoreConfigError, match="dense size 32"):
        other.ensure_collection(384)
