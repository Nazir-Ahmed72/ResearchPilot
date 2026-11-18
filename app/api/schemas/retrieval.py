from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.types import RetrievalStrategy
from app.retrieval.types import RetrievalResult


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    top_k: int | None = Field(default=None, ge=1, le=50)
    collection: str | None = Field(default=None, max_length=200)
    paper_ids: list[UUID] = Field(default_factory=list, max_length=100)
    sections: list[str] = Field(default_factory=list, max_length=50)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _year_range(self) -> SearchRequest:
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    paper_id: str
    title: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    text: str
    score: float = Field(description="Score of the final stage (rerank, rrf, dense or sparse)")
    scores: dict[str, float]
    ranks: dict[str, int]


class QueryTransformationOut(BaseModel):
    complexity: str
    rewritten: str | None
    alternatives: list[str]
    sub_questions: list[str]
    used_llm: bool
    error: str | None


class SearchResponse(BaseModel):
    query: str
    strategy: RetrievalStrategy
    queries: list[str]
    candidate_count: int
    timings_ms: dict[str, float]
    transformation: QueryTransformationOut | None
    results: list[SearchHit]

    @classmethod
    def from_result(cls, result: RetrievalResult) -> SearchResponse:
        t = result.transformation
        return cls(
            query=result.query,
            strategy=result.strategy,
            queries=result.queries,
            candidate_count=result.candidate_count,
            timings_ms=result.timings_ms,
            transformation=QueryTransformationOut(
                complexity=t.complexity,
                rewritten=t.rewritten,
                alternatives=t.alternatives,
                sub_questions=t.sub_questions,
                used_llm=t.used_llm,
                error=t.error,
            )
            if t
            else None,
            results=[
                SearchHit(
                    rank=i,
                    chunk_id=c.chunk_id,
                    paper_id=c.paper_id,
                    title=c.title,
                    section=c.section,
                    page_start=_int(c.payload.get("page_start")),
                    page_end=_int(c.payload.get("page_end")),
                    text=c.text,
                    score=round(c.score, 6),
                    scores={k: round(v, 6) for k, v in c.scores.items()},
                    ranks=c.ranks,
                )
                for i, c in enumerate(result.chunks, start=1)
            ],
        )


def _int(value: Any) -> int | None:
    return int(value) if value is not None else None
