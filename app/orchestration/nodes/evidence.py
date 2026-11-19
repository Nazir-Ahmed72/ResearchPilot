"""Evidence evaluation: is the selected evidence enough, and what is missing?

No evidence -> insufficient, deterministically (no LLM call). Otherwise the LLM judges coverage,
flags irrelevant passages (removed from the evidence) and proposes follow-up queries. If the LLM
call fails the run continues as if evidence were sufficient, with the error recorded, because
an extra retrieval round could not be planned without it anyway.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.logging import redact
from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.prompts import format_evidence
from app.orchestration.schemas import EvidenceAssessment
from app.orchestration.state import EvidenceItem, ResearchState


def evaluate_evidence(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    evidence: list[EvidenceItem] = state.get("evidence") or []
    analysis = state.get("analysis") or {}
    if not evidence:
        concepts = analysis.get("key_concepts") or []
        assessment = EvidenceAssessment(
            sufficient=False,
            missing_aspects=["no relevant passages were found"],
            follow_up_queries=concepts[:3],
        )
        return {
            "assessment": assessment.model_dump(),
            "_trace": {"sufficient": False, "source": "no-evidence"},
        }
    if deps.evaluator is None:
        return {
            "assessment": EvidenceAssessment(sufficient=True).model_dump(),
            "_trace": {"sufficient": True, "source": "no-evaluator"},
        }
    try:
        result = deps.evaluator.invoke(
            {
                "question": state["question"],
                "analysis_type": analysis.get("analysis_type", "factual"),
                "evidence": format_evidence(evidence, deps.config.evidence_chars),
            },
            config,
        )
        if not isinstance(result, EvidenceAssessment):
            raise TypeError(f"evaluator returned {type(result).__name__}")
    except Exception as exc:  # noqa: BLE001
        message = redact(f"evaluate_evidence: {type(exc).__name__}: {exc}")[:300]
        return {
            "assessment": EvidenceAssessment(sufficient=True).model_dump(),
            "errors": [message],
            "_trace": {"sufficient": True, "source": "fallback"},
        }

    valid_ids = {e["eid"] for e in evidence}
    irrelevant = [i for i in result.irrelevant_evidence_ids if i in valid_ids]
    kept = [e for e in evidence if e["eid"] not in irrelevant]
    sufficient = result.sufficient and bool(kept)
    assessment = {
        **result.model_dump(),
        "sufficient": sufficient,
        "irrelevant_evidence_ids": irrelevant,
    }
    # Re-number so citation ids stay dense (E1..En) after dropping irrelevant passages.
    renumbered = [EvidenceItem(**{**e, "eid": f"E{i}"}) for i, e in enumerate(kept, start=1)]
    return {
        "assessment": assessment,
        "evidence": renumbered,
        "_trace": {
            "sufficient": sufficient,
            "dropped": len(irrelevant),
            "missing": result.missing_aspects,
        },
    }
