"""Query understanding: intent, complexity, cross-paper need, key concepts.

LLM classification with a keyword-heuristic fallback, then deterministic normalisation: every
non-factual analysis type is treated as complex (it needs planning and evidence evaluation).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.logging import redact
from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.schemas import QueryAnalysis
from app.orchestration.state import ResearchState

logger = logging.getLogger(__name__)

_PATTERNS = [
    ("literature_review", r"\b(literature review|survey|overview of (the )?(research|work))\b"),
    (
        "research_gaps",
        r"\b(gaps?|open (problems?|questions?|challenges?)|future (work|directions?))\b",
    ),
    ("limitations", r"\b(limitations?|weakness(es)?|drawbacks?|shortcomings?|fail(s|ures?)?)\b"),
    ("comparison", r"\b(compare|comparison|versus|vs\.?|differ(ence|ences|s)?|contrast)\b"),
]


def heuristic_analysis(question: str) -> QueryAnalysis:
    lowered = question.lower()
    analysis_type = "factual"
    for name, pattern in _PATTERNS:
        if re.search(pattern, lowered):
            analysis_type = name
            break
    multi = analysis_type != "factual" or len(re.findall(r"\band\b|,", lowered)) >= 2
    return QueryAnalysis(
        analysis_type=analysis_type,  # type: ignore[arg-type]
        complexity="complex" if multi else "simple",
        requires_cross_paper=analysis_type in {"comparison", "research_gaps", "literature_review"},
        key_concepts=[],
    )


def understand_query(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    question = state["question"]
    errors: list[str] = []
    source = "llm"
    analysis: QueryAnalysis | None = None
    if deps.analyzer is not None:
        try:
            result = deps.analyzer.invoke({"question": question[:2000]}, config)
            analysis = result if isinstance(result, QueryAnalysis) else None
        except Exception as exc:  # noqa: BLE001 - classification is not worth failing the run
            errors.append(redact(f"understand_query: {type(exc).__name__}: {exc}")[:300])
    if analysis is None:
        analysis, source = heuristic_analysis(question), "heuristic"
    analysis = analysis.model_copy(deep=True)  # never mutate the object the LLM step returned

    override = state.get("analysis_type_override")
    if override:
        analysis.analysis_type = override  # type: ignore[assignment]
    if analysis.analysis_type != "factual":
        analysis.complexity = "complex"
    if analysis.analysis_type in {"comparison", "research_gaps", "literature_review"}:
        analysis.requires_cross_paper = True

    data = {**analysis.model_dump(), "source": source}
    logger.info("Query understood", extra={"analysis": data})
    return {
        "analysis": data,
        "errors": errors,
        "retry_count": 0,
        "repair_count": 0,
        "_trace": {
            "analysis_type": data["analysis_type"],
            "complexity": data["complexity"],
            "source": source,
        },
    }
