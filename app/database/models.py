"""SQLAlchemy ORM models for application metadata.

Design notes
------------
* Vectors live in Qdrant only; PostgreSQL stores metadata, job state and results.
* Primary keys are UUIDs so IDs can be generated before insert and used as stable Qdrant payload
  references across both stores.
* Status enums are stored as short strings (``native_enum=False``) so adding a new state later is
  a code change, not a ``ALTER TYPE`` migration, and the schema stays portable to SQLite for tests.
* JSON columns use ``JSONB`` on PostgreSQL and plain JSON elsewhere.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.types import AnalysisType, RetrievalStrategy

__all__ = ["AnalysisType", "RetrievalStrategy"]  # re-exported for existing imports

JSONType = JSON().with_variant(JSONB(), "postgresql")

# Deterministic constraint names make Alembic migrations reviewable and reversible.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}


def _str_enum(enum_cls: type[enum.Enum]) -> Enum:
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        validate_strings=True,
        values_callable=lambda members: [m.value for m in members],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# --- Enums ------------------------------------------------------------------------------------


class PaperStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # research run paused for human review (resumable)


# --- Tables -----------------------------------------------------------------------------------


class PaperCollection(TimestampMixin, Base):
    __tablename__ = "paper_collections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    papers: Mapped[list[Paper]] = relationship(back_populates="collection")


class Paper(TimestampMixin, Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("paper_collections.id", ondelete="SET NULL"), index=True
    )
    # SHA-256 of the raw PDF bytes: detects duplicate uploads and seeds deterministic chunk IDs.
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
    publication_year: Mapped[int | None] = mapped_column(Integer)
    doi: Mapped[str | None] = mapped_column(String(255))
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pdf_metadata: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    status: Mapped[PaperStatus] = mapped_column(
        _str_enum(PaperStatus), default=PaperStatus.PENDING, nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    collection: Mapped[PaperCollection | None] = relationship(back_populates="papers")
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def collection_name(self) -> str | None:
        return self.collection.name if self.collection else None


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        _str_enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True
    )
    task_id: Mapped[str | None] = mapped_column(String(255))
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    paper: Mapped[Paper] = relationship(back_populates="ingestion_jobs")


class ResearchRequest(TimestampMixin, Base):
    __tablename__ = "research_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("paper_collections.id", ondelete="SET NULL"), index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_type: Mapped[AnalysisType | None] = mapped_column(_str_enum(AnalysisType))
    retrieval_strategy: Mapped[RetrievalStrategy] = mapped_column(
        _str_enum(RetrievalStrategy), default=RetrievalStrategy.HYBRID_RERANK, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        _str_enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True
    )
    # LangGraph checkpoint thread; equals the request id by default but kept explicit.
    thread_id: Mapped[str | None] = mapped_column(String(255), index=True)
    # Graph node currently executing (progress for status polling).
    current_node: Mapped[str | None] = mapped_column(String(64))
    options: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    result: Mapped[ResearchResult | None] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    __table_args__ = (Index("ix_research_requests_created_at", "created_at"),)


class ResearchResult(TimestampMixin, Base):
    __tablename__ = "research_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_requests.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    findings: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    citations: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
    evidence_sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False)
    verification: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    # Latency, token usage, cost, retry count, node timings.
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    request: Mapped[ResearchRequest] = relationship(back_populates="result")


class EvaluationRun(TimestampMixin, Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    experiment: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # E1..E5
    retrieval_strategy: Mapped[RetrievalStrategy] = mapped_column(
        _str_enum(RetrievalStrategy), nullable=False
    )
    dataset_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _str_enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True
    )
    config: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    results_path: Mapped[str | None] = mapped_column(String(1024))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
