"""Citation verification.

1. Deterministic (always): every cited ID must exist in the evidence; every *reported* claim
   needs at least one valid citation. *Inferred* claims without citations are allowed but are
   labelled hypotheses.
2. LLM (if available): for each cited claim, does the cited text actually support it?

Problems trigger at most ``max_repair_attempts`` re-syntheses; whatever is still unsupported
afterwards is removed from the answer by the response node and reported separately.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.logging import redact
from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.prompts import format_evidence
from app.orchestration.schemas import VerificationResult
from app.orchestration.state import EvidenceItem, ResearchState

CITATION_MARKER = re.compile(r"\[(E\d+)\]")
PROBLEM_VERDICTS = {"unsupported", "uncited"}


def collect_claims(synthesis: dict[str, Any]) -> list[dict[str, Any]]:
    """Summary (C0) plus every section claim, in reading order."""
    summary = synthesis.get("summary", "")
    # Summary support = inline [E#] markers plus the structured summary_evidence_ids field
    # (models often omit inline markers; the structured field is more reliable).
    summary_ids = CITATION_MARKER.findall(summary) + list(
        synthesis.get("summary_evidence_ids") or []
    )
    claims = [
        {
            "claim_id": "C0",
            "section": None,
            "text": CITATION_MARKER.sub("", summary).strip(),
            "basis": "reported",
            "cited": list(dict.fromkeys(summary_ids)),
        }
    ]
    n = 1
    for section in synthesis.get("sections", []):
        for claim in section.get("claims", []):
            claims.append(
                {
                    "claim_id": f"C{n}",
                    "section": section.get("heading"),
                    "text": claim.get("text", ""),
                    "basis": claim.get("basis", "reported"),
                    "cited": list(dict.fromkeys(claim.get("evidence_ids") or [])),
                }
            )
            n += 1
    return claims


def _claims_prompt(claims: list[dict[str, Any]], by_id: dict[str, EvidenceItem], chars: int) -> str:
    """Each cited passage appears ONCE, followed by the claims that reference it by id.

    Repeating a passage for every claim that cites it made a 10-claim comparison exceed Groq's
    8k tokens-per-minute free-tier limit (10.4k tokens); de-duplication roughly halves it.
    """
    cited_ids = list(dict.fromkeys(e for c in claims for e in c["valid_citations"]))
    evidence = format_evidence([by_id[e] for e in cited_ids], chars)
    lines = [
        f"Claim {c['claim_id']} (cites {', '.join(c['valid_citations'])}): {c['text']}"
        for c in claims
    ]
    return f"Evidence:\n{evidence}\n\nClaims:\n" + "\n".join(lines)


def verify_citations(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    by_id = {e["eid"]: e for e in evidence}
    claims = collect_claims(state.get("synthesis") or {})

    for c in claims:
        c["valid_citations"] = [e for e in c["cited"] if e in by_id]
        c["invalid_citations"] = [e for e in c["cited"] if e not in by_id]
        c["reason"] = ""
        if c["valid_citations"]:
            c["verdict"] = "unverified"  # pending the LLM check
        elif c["basis"] == "inferred":
            c["verdict"] = "hypothesis"
        else:
            c["verdict"] = "uncited"
            c["reason"] = "reported claim without a valid citation"

    to_check = [c for c in claims if c["verdict"] == "unverified"]
    errors: list[str] = []
    llm_checked = False
    if to_check and deps.verifier is not None:
        try:
            result = deps.verifier.invoke(
                {"claims": _claims_prompt(to_check, by_id, deps.config.evidence_chars)}, config
            )
            if not isinstance(result, VerificationResult):
                raise TypeError(f"verifier returned {type(result).__name__}")
            verdicts = {v.claim_id: v for v in result.verdicts}
            for c in to_check:
                v = verdicts.get(c["claim_id"])
                if v is not None:
                    c["verdict"], c["reason"] = v.verdict, v.reason
            llm_checked = True
        except Exception as exc:  # noqa: BLE001 - keep deterministic results, flag as unverified
            errors.append(redact(f"verify_citations: {type(exc).__name__}: {exc}")[:300])

    problems = []
    for c in claims:
        if c["verdict"] in PROBLEM_VERDICTS:
            problems.append(f'{c["claim_id"]} "{c["text"][:160]}": {c["reason"] or c["verdict"]}')
        if c["invalid_citations"]:
            problems.append(f"{c['claim_id']} cites unknown evidence {c['invalid_citations']}")

    metrics = verification_metrics(claims)
    needs_repair = bool(problems) and state.get("repair_count", 0) < deps.config.max_repair_attempts
    return {
        "verification": {
            "claims": claims,
            "metrics": metrics,
            "problems": problems,
            "needs_repair": needs_repair,
            "llm_checked": llm_checked,
        },
        "errors": errors,
        "_trace": {"problems": len(problems), "needs_repair": needs_repair, **metrics},
    }


def verification_metrics(claims: list[dict[str, Any]]) -> dict[str, Any]:
    total_cites = sum(len(c["cited"]) for c in claims)
    valid_cites = sum(len(c["valid_citations"]) for c in claims)
    reported = [c for c in claims if c["basis"] == "reported"]
    cited_reported = [c for c in reported if c["valid_citations"]]
    checked = [
        c for c in claims if c["verdict"] in {"supported", "partially_supported", "unsupported"}
    ]
    supported = sum(c["verdict"] == "supported" for c in checked)
    partial = sum(c["verdict"] == "partially_supported" for c in checked)

    def ratio(num: float, den: int) -> float | None:
        return round(num / den, 3) if den else None

    return {
        "claims": len(claims),
        "citations": total_cites,
        "citation_validity": ratio(valid_cites, total_cites),
        "citation_coverage": ratio(len(cited_reported), len(reported)),
        "checked_claims": len(checked),
        "supported": supported,
        "partially_supported": partial,
        "unsupported": sum(c["verdict"] == "unsupported" for c in checked),
        "support_rate": ratio(supported + 0.5 * partial, len(checked)),
    }
