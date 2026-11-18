from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.database.models import JobStatus, Paper, PaperStatus


class UploadResult(BaseModel):
    filename: str
    status: str = Field(description="accepted | duplicate | rejected | failed")
    paper_id: UUID | None = None
    job_id: UUID | None = None
    error_code: str | None = None
    message: str | None = None


class UploadResponse(BaseModel):
    accepted: int
    results: list[UploadResult]


class IngestionJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: JobStatus
    attempts: int
    task_id: str | None
    error_message: str | None
    stats: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class PaperSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    authors: list[str]
    publication_year: int | None
    original_filename: str
    status: PaperStatus
    page_count: int | None
    chunk_count: int
    collection: str | None = Field(
        default=None, validation_alias=AliasChoices("collection_name", "collection")
    )
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_paper(cls, paper: Paper) -> PaperSummary:
        return cls.model_validate(paper)


class PaperDetail(PaperSummary):
    doi: str | None
    file_size_bytes: int
    content_hash: str
    error_message: str | None
    pdf_metadata: dict[str, Any]
    ingestion_jobs: list[IngestionJobOut]

    @classmethod
    def from_paper(cls, paper: Paper) -> PaperDetail:
        out = cls.model_validate(paper)
        out.ingestion_jobs = sorted(out.ingestion_jobs, key=lambda j: j.created_at, reverse=True)
        return out


class PaperList(BaseModel):
    items: list[PaperSummary]
    total: int
    limit: int
    offset: int


class ReingestResponse(BaseModel):
    paper_id: UUID
    job_id: UUID
    status: JobStatus
