"""Retrieval data types, independent of Qdrant and of any model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from app.core.types import RetrievalStrategy


@dataclass(frozen=True)
class RetrievalFilters:
    """Restricts search to a subset of the corpus. All conditions are ANDed."""

    collection_id: str | None = None
    paper_ids: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None

    @property
    def is_empty(self) -> bool:
        return not (
            self.collection_id or self.paper_ids or self.sections or self.year_from or self.year_to
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk with its provenance and the scores/ranks it received at each stage.

    ``score`` is the score of the stage that produced the current ordering ('dense', 'sparse',
    'rrf' or 'rerank'); ``scores``/``ranks`` keep the whole history for analysis.
    """

    chunk_id: str
    payload: dict[str, Any]
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def paper_id(self) -> str:
        return str(self.payload.get("paper_id", ""))

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def section(self) -> str | None:
        return self.payload.get("section")

    @property
    def title(self) -> str | None:
        return self.payload.get("title")

    def with_stage(self, stage: str, score: float, rank: int) -> RetrievedChunk:
        return replace(
            self,
            score=score,
            scores={**self.scores, stage: score},
            ranks={**self.ranks, stage: rank},
        )


@dataclass
class TransformedQuery:
    original: str
    queries: list[str]  # what is actually searched; always starts with the original
    complexity: str = "simple"
    rewritten: str | None = None
    alternatives: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    used_llm: bool = False
    error: str | None = None


@dataclass
class RetrievalResult:
    query: str
    strategy: RetrievalStrategy
    chunks: list[RetrievedChunk]
    queries: list[str]
    candidate_count: int
    timings_ms: dict[str, float]
    transformation: TransformedQuery | None = None
