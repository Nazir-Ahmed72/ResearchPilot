"""Structured outputs requested from the LLM. Every list is trimmed (not rejected) to its cap,
so a slightly over-eager model response is still usable; everything is validated by Pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AnalysisTypeLiteral = Literal[
    "factual", "comparison", "limitations", "research_gaps", "literature_review"
]


def _trim(values: list[str], cap: int) -> list[str]:
    out = []
    for v in values:
        v = " ".join(str(v).split())
        if v and v not in out:
            out.append(v)
    return out[:cap]


class QueryAnalysis(BaseModel):
    analysis_type: AnalysisTypeLiteral = Field(
        description="factual: explain/define something. comparison: compare approaches/papers. "
        "limitations: what limitations are reported. research_gaps: open problems, future "
        "directions. literature_review: a structured review of the topic."
    )
    complexity: Literal["simple", "complex"] = Field(
        description="simple: ONE focused fact answerable from one passage. complex: several "
        "parts (e.g. 'what is X and how does it work'), several entities or papers, or "
        "multi-hop (the answer to one part is needed to look up another)."
    )
    requires_cross_paper: bool = Field(
        description="True if answering needs evidence from more than one paper."
    )
    key_concepts: list[str] = Field(
        default_factory=list, description="Up to 6 key technical concepts named in the question."
    )

    @field_validator("key_concepts")
    @classmethod
    def _cap(cls, v: list[str]) -> list[str]:
        return _trim(v, 6)


class EvidenceAssessment(BaseModel):
    sufficient: bool = Field(
        description="True if the evidence is enough to answer every part of the question."
    )
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(
        default_factory=list, description="Parts of the question the evidence does not cover."
    )
    irrelevant_evidence_ids: list[str] = Field(
        default_factory=list, description="IDs (e.g. E3) of evidence unrelated to the question."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 short search queries that could find the missing information.",
    )

    @field_validator("covered_aspects", "missing_aspects")
    @classmethod
    def _cap_aspects(cls, v: list[str]) -> list[str]:
        return _trim(v, 8)

    @field_validator("irrelevant_evidence_ids")
    @classmethod
    def _cap_ids(cls, v: list[str]) -> list[str]:
        return _trim(v, 50)

    @field_validator("follow_up_queries")
    @classmethod
    def _cap_queries(cls, v: list[str]) -> list[str]:
        return _trim(v, 3)


class Claim(BaseModel):
    text: str = Field(description="One self-contained statement. No citation markers inside.")
    evidence_ids: list[str] = Field(
        default_factory=list, description="IDs of the evidence passages that support it."
    )
    # A plain string (not an enum) on purpose: providers that validate tool calls server-side
    # (Groq) reject the WHOLE answer if the model writes e.g. "uncertainty" here. Unknown values
    # are normalised below instead of failing the run.
    basis: str = Field(
        default="reported",
        description="Exactly 'reported' (stated in the cited evidence) or 'inferred' (your own "
        "synthesis or hypothesis drawn from the evidence, e.g. a possible research gap).",
    )

    @field_validator("basis", mode="before")
    @classmethod
    def _normalise_basis(cls, value: object) -> str:
        text = str(value or "reported").strip().lower()
        return text if text in {"reported", "inferred"} else "inferred"


class SynthesisSection(BaseModel):
    heading: str
    claims: list[Claim] = Field(default_factory=list)


class Synthesis(BaseModel):
    # Decided FIRST (models fill fields in order): does the evidence answer the question at all?
    evidence_answers_question: bool = Field(
        default=True,
        description="Decide this first. True if the evidence answers the question at least "
        "partly (then answer that part and put what is missing in 'uncertainty'). False only "
        "if the passages contain nothing relevant to what is asked, e.g. they never mention the "
        "model, method or benchmark the question is about.",
    )
    summary: str = Field(
        description="A direct 2-5 sentence answer. Cite evidence inline as [E1], [E2]. If the "
        "evidence does not answer the question, say only that."
    )
    summary_evidence_ids: list[str] = Field(
        default_factory=list, description="IDs of the evidence passages the summary relies on."
    )
    sections: list[SynthesisSection] = Field(default_factory=list)
    uncertainty: str = Field(
        default="",
        description="What the evidence does not establish, conflicts, and caveats.",
    )


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str = ""


class VerificationResult(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
