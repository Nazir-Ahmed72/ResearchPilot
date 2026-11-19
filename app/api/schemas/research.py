from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.core.types import AnalysisType, RetrievalStrategy
from app.database.models import JobStatus, ResearchRequest


class ResearchQuery(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    analysis_type: AnalysisType | None = Field(
        default=None, description="Force an analysis mode; otherwise it is inferred."
    )
    strategy: RetrievalStrategy | None = Field(
        default=None, description="Retrieval strategy (default: RESEARCH_DEFAULT_STRATEGY)."
    )
    collection: str | None = Field(default=None, max_length=200)
    paper_ids: list[UUID] = Field(default_factory=list, max_length=50)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 5:
            raise ValueError("question is too short")
        return value


class ResearchAccepted(BaseModel):
    request_id: UUID
    status: JobStatus
    status_url: str


class Reference(BaseModel):
    number: int
    paper_id: str
    chunk_id: str
    title: str | None
    year: int | None
    section: str | None
    page_start: int | None
    page_end: int | None
    label: str


class ResearchResultOut(BaseModel):
    answer_markdown: str
    summary: str | None
    sections: list[dict[str, Any]]
    references: list[Reference]
    evidence_sufficient: bool
    uncertainty: str | None
    removed_statements: list[dict[str, Any]]
    queries: list[str]
    verification: dict[str, Any] | None
    metrics: dict[str, Any]


class ResearchStatus(BaseModel):
    request_id: UUID
    status: JobStatus
    current_node: str | None
    question: str
    analysis_type: AnalysisType | None
    strategy: RetrievalStrategy
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: ResearchResultOut | None = None

    @classmethod
    def from_request(cls, request: ResearchRequest) -> ResearchStatus:
        result = None
        if request.result is not None:
            r = request.result
            findings = r.findings or {}
            result = ResearchResultOut(
                answer_markdown=r.answer,
                summary=findings.get("summary"),
                sections=findings.get("sections", []),
                references=[Reference(**ref) for ref in r.citations or []],
                evidence_sufficient=r.evidence_sufficient,
                uncertainty=findings.get("uncertainty"),
                removed_statements=findings.get("removed_statements", []),
                queries=findings.get("queries", []),
                verification=(r.verification or {}).get("metrics"),
                metrics={k: v for k, v in (r.metrics or {}).items() if k != "trace"},
            )
        return cls(
            request_id=request.id,
            status=request.status,
            current_node=request.current_node,
            question=request.question,
            analysis_type=request.analysis_type,
            strategy=request.retrieval_strategy,
            error_message=request.error_message,
            created_at=request.created_at,
            started_at=request.started_at,
            finished_at=request.finished_at,
            result=result,
        )
