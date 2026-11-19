from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import RetrievalStrategy
from app.database.models import JobStatus


class EvaluationRunRequest(BaseModel):
    name: str = Field(default="api-eval", min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    dataset: str = Field(default="rag_papers_v1", pattern=r"^[\w.-]+$")
    experiments: list[str] = Field(default_factory=lambda: ["E1", "E2", "E3", "E4", "E5"])
    top_k: int = Field(default=5, ge=1, le=50)
    include_generation: bool = False
    generation_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    question_ids: list[str] | None = Field(default=None, max_length=200)
    limit: int | None = Field(default=None, ge=1, le=500)
    ragas: bool = False
    pace_seconds: float = Field(default=0.0, ge=0, le=300)
    collection: str | None = Field(default=None, max_length=200)
    langsmith: bool = False


class EvaluationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    experiment: str
    retrieval_strategy: RetrievalStrategy
    dataset_name: str
    status: JobStatus
    metrics: dict[str, Any]
    results_path: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EvaluationAccepted(BaseModel):
    batch_id: UUID
    runs: list[EvaluationRunOut]


class EvaluationResults(BaseModel):
    items: list[EvaluationRunOut]
