"""ORM model behaviour against a real (SQLite) database."""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.core.resources import Resources
from app.database.models import (
    IngestionJob,
    JobStatus,
    Paper,
    PaperCollection,
    PaperStatus,
    ResearchRequest,
    ResearchResult,
)


@pytest.fixture
def session(resources: Resources) -> Session:
    with resources.session_factory() as s:
        yield s


def _paper(content: bytes = b"pdf", **kw: object) -> Paper:
    defaults = dict(
        content_hash=hashlib.sha256(content).hexdigest(),
        original_filename="paper.pdf",
        storage_path="uploads/x.pdf",
        file_size_bytes=len(content),
    )
    defaults.update(kw)
    return Paper(**defaults)


def test_paper_defaults_and_timestamps(session: Session) -> None:
    paper = _paper(title="Attention Is All You Need", authors=["Vaswani"])
    session.add(paper)
    session.commit()
    loaded = session.get(Paper, paper.id)
    assert isinstance(loaded.id, uuid.UUID)
    assert loaded.status is PaperStatus.PENDING
    assert loaded.authors == ["Vaswani"]
    assert loaded.pdf_metadata == {}
    assert loaded.chunk_count == 0
    assert loaded.created_at is not None and loaded.updated_at is not None


def test_duplicate_content_hash_rejected(session: Session) -> None:
    session.add(_paper(b"same"))
    session.commit()
    session.add(_paper(b"same", original_filename="copy.pdf"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_invalid_status_value_rejected(session: Session) -> None:
    session.add(_paper(status="exploded"))
    with pytest.raises(StatementError):
        session.commit()


def test_deleting_paper_cascades_to_ingestion_jobs(session: Session) -> None:
    paper = _paper()
    paper.ingestion_jobs.append(IngestionJob(status=JobStatus.SUCCEEDED))
    session.add(paper)
    session.commit()
    session.delete(paper)
    session.commit()
    assert session.scalars(select(IngestionJob)).all() == []


def test_deleting_collection_keeps_papers(session: Session) -> None:
    collection = PaperCollection(name="agents")
    paper = _paper(collection=collection)
    session.add_all([collection, paper])
    session.commit()
    session.delete(collection)
    session.commit()
    session.expire_all()
    assert session.get(Paper, paper.id).collection_id is None


def test_research_request_result_one_to_one(session: Session) -> None:
    request = ResearchRequest(question="What is RAG?")
    request.result = ResearchResult(
        answer="...", evidence_sufficient=True, citations=[{"paper_id": "p", "page": 1}]
    )
    session.add(request)
    session.commit()
    session.add(ResearchResult(request_id=request.id, answer="dup", evidence_sufficient=False))
    with pytest.raises(IntegrityError):
        session.commit()
