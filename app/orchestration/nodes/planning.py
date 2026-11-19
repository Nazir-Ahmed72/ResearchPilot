"""Query planning: which searches to run.

First pass: the LLM query transformer (rewrite / decomposition), plus deterministic queries that
target the analysis type (e.g. '<concept> limitations' for a limitations question).
Retry pass (after insufficient evidence): the evaluator's follow-up queries, or the missing
aspects themselves when it suggested none. ``retry_count`` is incremented here, so the bound on
plan -> retrieve loops is enforced in one place.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.state import ResearchState
from app.retrieval.query_transformer import QueryPlan, build_queries

_TYPE_SUFFIX = {
    "limitations": "limitations",
    "research_gaps": "limitations and future work",
    "literature_review": "overview",
}


def _typed_queries(analysis: dict[str, Any]) -> list[str]:
    suffix = _TYPE_SUFFIX.get(analysis.get("analysis_type", ""))
    concepts = analysis.get("key_concepts") or []
    return [f"{c} {suffix}" for c in concepts[:2]] if suffix else []


def plan_queries(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    question = state["question"]
    analysis = state.get("analysis") or {}
    max_q = deps.transformer.max_queries
    is_retry = state.get("assessment") is not None and not state["assessment"].get("sufficient")

    if is_retry:
        # The router only sends us here when the evaluator proposed follow-up queries.
        follow_ups = (state["assessment"] or {}).get("follow_up_queries") or []
        plan = QueryPlan(complexity="complex", rewritten_query=question, sub_questions=follow_ups)
        queries = build_queries(question, plan, max_q)[1:] or [question]  # new angles only
        retry_count = state.get("retry_count", 0) + 1
        return {
            "search_queries": queries,
            "retry_count": retry_count,
            "_trace": {"retry": retry_count, "queries": queries},
        }

    transformed = deps.transformer.transform(question)
    extra = _typed_queries(analysis)
    plan = QueryPlan(
        complexity="complex",
        rewritten_query=transformed.rewritten or question,
        sub_questions=transformed.sub_questions + extra,
        alternative_queries=transformed.alternatives,
    )
    queries = build_queries(question, plan, max_q + len(extra))
    errors = [f"plan_queries: {transformed.error}"] if transformed.error else []
    return {
        "search_queries": queries,
        "errors": errors,
        "_trace": {"queries": queries, "used_llm": transformed.used_llm},
    }
