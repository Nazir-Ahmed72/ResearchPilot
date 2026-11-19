"""LangGraph state for the research workflow.

The state is a ``TypedDict`` of JSON-serialisable values (plain dicts/lists, not ORM objects or
models), so every checkpoint written to PostgreSQL is small, portable and human-readable. No PDF
bytes or vectors are kept here: only chunk IDs, provenance and the chunk text needed for
synthesis.

Fields updated by several nodes over time use reducers:
* ``errors`` / ``trace`` append,
* ``node_timings`` sums durations when a node runs more than once (retries, repairs).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def sum_timings(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    merged = dict(left or {})
    for node, ms in (right or {}).items():
        merged[node] = round(merged.get(node, 0.0) + ms, 1)
    return merged


class EvidenceItem(TypedDict):
    eid: str  # short citation id shown to the LLM ("E1", "E2", ...)
    chunk_id: str
    paper_id: str
    title: str | None
    year: int | None
    section: str | None
    page_start: int | None
    page_end: int | None
    text: str
    score: float  # final-stage score (rerank if enabled, else fusion)
    scores: dict[str, float]


class ResearchState(TypedDict, total=False):
    # --- input -----------------------------------------------------------------------------
    request_id: str
    question: str
    analysis_type_override: str | None
    strategy: str  # RetrievalStrategy value
    filters: dict[str, Any]  # RetrievalFilters fields

    # --- query understanding / planning ----------------------------------------------------
    analysis: dict[str, Any]  # QueryAnalysis
    search_queries: list[str]

    # --- retrieval -------------------------------------------------------------------------
    candidates: list[EvidenceItem]  # latest retrieval round, before selection
    evidence: list[EvidenceItem]  # selected evidence (accumulates across retries)
    assessment: dict[str, Any] | None  # EvidenceAssessment
    retry_count: int

    # --- synthesis / verification ----------------------------------------------------------
    synthesis: dict[str, Any] | None  # Synthesis
    repair_count: int
    repair_feedback: list[str]
    verification: dict[str, Any] | None
    final: dict[str, Any] | None  # rendered answer

    # --- execution metadata ----------------------------------------------------------------
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    node_timings: Annotated[dict[str, float], sum_timings]
