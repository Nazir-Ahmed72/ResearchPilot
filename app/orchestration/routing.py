"""Conditional edges. Pure functions of the state, so routing is trivially unit-testable.

Loops are bounded by counters that the *nodes* increment (``retry_count`` in plan_queries,
``repair_count`` in synthesize) and that these functions compare with configured maxima.
"""

from __future__ import annotations

from typing import Literal

from app.core.types import RetrievalStrategy
from app.orchestration.state import ResearchState


def _complex(state: ResearchState) -> bool:
    return (state.get("analysis") or {}).get("complexity") == "complex"


def route_after_understanding(state: ResearchState) -> Literal["plan_queries", "retrieve"]:
    """Simple questions go straight to retrieval; complex ones (and the query-transformation
    strategy, E5) are planned first."""
    if _complex(state) or state.get("strategy") == RetrievalStrategy.HYBRID_QT_RERANK.value:
        return "plan_queries"
    return "retrieve"


def route_after_rerank(
    state: ResearchState,
) -> Literal["evaluate_evidence", "synthesize", "respond"]:
    if _complex(state):
        return "evaluate_evidence"
    return "synthesize" if state.get("evidence") else "respond"


def route_after_evaluation(
    state: ResearchState, max_retries: int
) -> Literal["plan_queries", "synthesize", "respond"]:
    assessment = state.get("assessment") or {}
    can_retry = state.get("retry_count", 0) < max_retries
    if not assessment.get("sufficient") and can_retry and assessment.get("follow_up_queries"):
        return "plan_queries"
    return "synthesize" if state.get("evidence") else "respond"


def route_after_verification(state: ResearchState) -> Literal["synthesize", "respond"]:
    return "synthesize" if (state.get("verification") or {}).get("needs_repair") else "respond"
