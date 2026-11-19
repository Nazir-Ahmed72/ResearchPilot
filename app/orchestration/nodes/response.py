"""Response generation: renders the verified synthesis into the final answer.

Deliberately deterministic (no LLM): the final text is assembled only from claims that passed
verification, so this last step cannot introduce new, unverified statements. Evidence IDs are
turned into numbered references [1], [2] in order of first use, each pointing to paper, section
and pages.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.nodes.verification import CITATION_MARKER
from app.orchestration.state import EvidenceItem, ResearchState

NO_EVIDENCE_MESSAGE = (
    "No passages relevant to this question were found in the selected papers, so no answer "
    "can be given from the collection. Try rephrasing the question, widening the paper or "
    "collection filter, or uploading papers that cover this topic."
)
DROPPED_VERDICTS = {"unsupported", "uncited"}
_ABSENCE = re.compile(
    r"\b(evidence|passages?|papers?|sources?|documents?|context)\b[^.]{0,40}\b"
    r"(does|do|did)\s+not\s+(provide|state|mention|contain|specify|include|report|discuss|"
    r"describe|give|address)\b",
    re.IGNORECASE,
)


def is_absence_statement(text: str) -> bool:
    """'The evidence does not mention X' style statements: uncertainty, not findings."""
    return bool(_ABSENCE.search(text))


class _References:
    def __init__(self, by_id: dict[str, EvidenceItem]) -> None:
        self.by_id = by_id
        self.numbers: dict[str, int] = {}

    def cite(self, eids: list[str]) -> str:
        nums = []
        for eid in eids:
            if eid not in self.by_id:
                continue
            if eid not in self.numbers:
                self.numbers[eid] = len(self.numbers) + 1
            nums.append(self.numbers[eid])
        return "".join(f"[{n}]" for n in sorted(set(nums)))

    def entries(self) -> list[dict[str, Any]]:
        out = []
        for eid, n in sorted(self.numbers.items(), key=lambda kv: kv[1]):
            e = self.by_id[eid]
            pages = (
                f"p. {e['page_start']}"
                if e["page_start"] == e["page_end"]
                else f"pp. {e['page_start']}-{e['page_end']}"
            )
            out.append(
                {
                    "number": n,
                    "paper_id": e["paper_id"],
                    "chunk_id": e["chunk_id"],
                    "title": e["title"],
                    "year": e["year"],
                    "section": e["section"],
                    "page_start": e["page_start"],
                    "page_end": e["page_end"],
                    "label": f"{e['title'] or 'Untitled'} ({e['year'] or 'n.d.'}), "
                    f"{e['section'] or 'unknown section'}, {pages}",
                }
            )
        return out


def _render_summary(summary: str, refs: _References) -> str:
    text = CITATION_MARKER.sub(lambda m: refs.cite([m.group(1)]), summary)
    return re.sub(r"\s+([.,;:])", r"\1", text).strip()


def respond(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    analysis = state.get("analysis") or {}
    assessment = state.get("assessment") or {}
    base = {
        "analysis_type": analysis.get("analysis_type"),
        "queries": state.get("search_queries") or [state["question"]],
    }

    if not evidence or not state.get("synthesis"):
        final = {
            **base,
            "answer_markdown": NO_EVIDENCE_MESSAGE,
            "summary": NO_EVIDENCE_MESSAGE,
            "sections": [],
            "references": [],
            "evidence_sufficient": False,
            "uncertainty": NO_EVIDENCE_MESSAGE,
            "removed_statements": [],
            "verification": None,
        }
        return {"final": final, "_trace": {"outcome": "no_evidence"}}

    synthesis = state["synthesis"] or {}
    verification = state.get("verification") or {}
    verdicts = {c["claim_id"]: c for c in verification.get("claims", [])}
    refs = _References({e["eid"]: e for e in evidence})

    summary = _render_summary(synthesis.get("summary", ""), refs)
    sections, removed, absences = [], [], []
    claim_no = 0
    for section in synthesis.get("sections", []):
        rendered = []
        for claim in section.get("claims", []):
            claim_no += 1
            v = verdicts.get(f"C{claim_no}", {})
            verdict = v.get("verdict", "unverified")
            if is_absence_statement(claim.get("text") or ""):
                # "The evidence does not mention X" is not a finding; it is uncertainty.
                absences.append(claim.get("text"))
                continue
            if verdict in DROPPED_VERDICTS:
                removed.append({"text": claim.get("text"), "reason": v.get("reason") or verdict})
                continue
            rendered.append(
                {
                    "text": claim.get("text"),
                    "basis": claim.get("basis"),
                    "verdict": verdict,
                    "citations": refs.cite(claim.get("evidence_ids") or []),
                }
            )
        if rendered:
            sections.append({"heading": section.get("heading"), "claims": rendered})

    # Two independent signals: the evaluator (complex path only) and the synthesizer, which
    # also runs on the simple path and can tell that the retrieved passages miss the question.
    # Three signals: the evaluator (complex path only), the synthesizer's own judgement, and a
    # deterministic check that at least one substantive (non-absence) claim survived.
    has_findings = any(s["claims"] for s in sections)
    sufficient = (
        bool(assessment.get("sufficient", True))
        and bool(synthesis.get("evidence_answers_question", True))
        and has_findings
    )
    uncertainty = " ".join(
        part for part in [(synthesis.get("uncertainty") or "").strip(), *absences] if part
    ).strip()
    lines = []
    if not sufficient:
        missing = "; ".join(assessment.get("missing_aspects") or [])
        lines.append(
            f"> **Evidence was insufficient** for parts of this question"
            f"{': ' + missing if missing else ''}. The answer covers only what the "
            f"retrieved passages support.\n"
        )
    lines += ["## Answer", summary, ""]
    for s in sections:
        lines.append(f"## {s['heading']}")
        for c in s["claims"]:
            prefix = "*Inferred:* " if c["basis"] == "inferred" else ""
            suffix = " *(partially supported)*" if c["verdict"] == "partially_supported" else ""
            suffix += " *(not verified)*" if c["verdict"] == "unverified" else ""
            lines.append(f"- {prefix}{c['text']} {c['citations']}{suffix}".rstrip())
        lines.append("")
    if uncertainty:
        lines += ["## Limitations and uncertainty", uncertainty, ""]
    if removed:
        lines += [
            "## Statements removed during verification",
            *[f"- {r['text']} ({r['reason']})" for r in removed],
            "",
        ]
    references = refs.entries()
    lines += ["## References", *[f"[{r['number']}] {r['label']}" for r in references]]

    final = {
        **base,
        "answer_markdown": "\n".join(lines).strip(),
        "summary": summary,
        "sections": sections,
        "references": references,
        "evidence_sufficient": sufficient,
        "uncertainty": uncertainty,
        "removed_statements": removed,
        "verification": verification.get("metrics"),
    }
    return {
        "final": final,
        "_trace": {"outcome": "answered", "references": len(references), "removed": len(removed)},
    }
