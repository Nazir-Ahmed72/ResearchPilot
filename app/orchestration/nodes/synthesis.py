"""Research synthesis: one structured LLM call over the selected evidence.

On a repair pass (citation verification found problems), the previous problems are passed back
so the model can drop or re-cite those claims. ``repair_count`` is incremented here, bounding the
synthesize <-> verify loop.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.prompts import ANALYSIS_GUIDANCE, format_evidence
from app.orchestration.schemas import Synthesis
from app.orchestration.state import ResearchState


def synthesize(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    analysis = state.get("analysis") or {}
    analysis_type = analysis.get("analysis_type", "factual")
    assessment = state.get("assessment") or {}
    verification = state.get("verification") or {}
    repairing = bool(verification.get("needs_repair"))

    sufficiency_note = ""
    if assessment and not assessment.get("sufficient", True):
        missing = "; ".join(assessment.get("missing_aspects") or []) or "unspecified"
        sufficiency_note = (
            "NOTE: the evidence was judged INSUFFICIENT. Missing: " + missing + ". Answer only "
            "what the evidence supports and state clearly what cannot be answered.\n\n"
        )
    repair_note = ""
    if repairing:
        problems = "\n".join(f"- {p}" for p in verification.get("problems", []))
        repair_note = (
            "Your previous answer had citation problems. Remove these claims or cite evidence "
            "that actually supports them:\n" + problems + "\n\n"
        )

    result = deps.synthesizer.invoke(
        {
            "question": state["question"],
            "analysis_type": analysis_type,
            "guidance": ANALYSIS_GUIDANCE.get(analysis_type, ANALYSIS_GUIDANCE["factual"]),
            "sufficiency_note": sufficiency_note,
            "repair_note": repair_note,
            "evidence": format_evidence(state.get("evidence") or [], deps.config.evidence_chars),
        },
        config,
    )
    if not isinstance(result, Synthesis):
        raise TypeError(f"synthesizer returned {type(result).__name__}, expected Synthesis")

    repair_count = state.get("repair_count", 0) + (1 if repairing else 0)
    claims = sum(len(s.claims) for s in result.sections)
    return {
        "synthesis": result.model_dump(),
        "repair_count": repair_count,
        "_trace": {"sections": len(result.sections), "claims": claims, "repair": repairing},
    }
