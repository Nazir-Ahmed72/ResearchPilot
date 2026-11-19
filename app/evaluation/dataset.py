"""Evaluation dataset: questions, reference answers and evidence labels.

Evidence is labelled by *content*, not by chunk id: an evidence item names a paper (matched by a
title fragment) and phrases the supporting passage must contain. A retrieved chunk is relevant to
an item if it comes from that paper and contains every phrase ("a|b" = either alternative;
case- and whitespace-insensitive). Labels therefore survive re-chunking and re-ingestion, and
``validate`` checks every label against the actual indexed chunks before a run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DATASETS_DIR = Path(__file__).parent / "datasets"
Category = Literal[
    "simple_factual",
    "comparison",
    "multi_document",
    "limitations",
    "multi_query",
    "insufficient_evidence",
]


class PaperRef(BaseModel):
    title_contains: str
    arxiv: str | None = None


class EvidenceLabel(BaseModel):
    paper: str
    contains: list[str] = Field(min_length=1)


class EvalQuestion(BaseModel):
    id: str
    category: Category
    question: str
    reference_answer: str = ""
    evidence: list[EvidenceLabel] = Field(default_factory=list)
    answerable: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> EvalQuestion:
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable questions need evidence labels")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable questions must not have evidence")
        return self


class EvalDataset(BaseModel):
    name: str
    description: str = ""
    papers: dict[str, PaperRef]
    questions: list[EvalQuestion]

    @model_validator(mode="after")
    def _papers_exist(self) -> EvalDataset:
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question ids")
        for q in self.questions:
            for e in q.evidence:
                if e.paper not in self.papers:
                    raise ValueError(f"{q.id}: unknown paper key {e.paper!r}")
        return self

    def subset(self, ids: Iterable[str] | None = None, limit: int | None = None) -> EvalDataset:
        questions = self.questions
        if ids:
            wanted = set(ids)
            questions = [q for q in questions if q.id in wanted]
        if limit:
            questions = questions[:limit]
        return self.model_copy(update={"questions": questions})


def load_dataset(name_or_path: str = "rag_papers_v1") -> EvalDataset:
    path = Path(name_or_path)
    if not path.suffix:
        path = DATASETS_DIR / f"{name_or_path}.json"
    return EvalDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def matches_paper(payload: Mapping[str, Any], ref: PaperRef) -> bool:
    return ref.title_contains.lower() in _norm(payload.get("title") or "")


def matches_label(
    payload: Mapping[str, Any], label: EvidenceLabel, papers: dict[str, PaperRef]
) -> bool:
    if not matches_paper(payload, papers[label.paper]):
        return False
    text = _norm(payload.get("text") or "")
    return all(any(_norm(alt) in text for alt in phrase.split("|")) for phrase in label.contains)


def relevance(
    payloads: list[Mapping[str, Any]], question: EvalQuestion, papers: dict[str, PaperRef]
) -> list[set[int]]:
    """For each retrieved chunk (in rank order), the indices of evidence labels it satisfies."""
    return [
        {i for i, label in enumerate(question.evidence) if matches_label(p, label, papers)}
        for p in payloads
    ]


class LabelCheck(BaseModel):
    question_id: str
    label_index: int
    paper: str
    contains: list[str]
    matching_chunks: int
    pages: list[int]


def validate(
    dataset: EvalDataset, chunks: list[Mapping[str, Any]], max_matches: int = 4
) -> list[LabelCheck]:
    """Check every label against the indexed chunks. Raises if a label matches no chunk (it can
    never be retrieved) or too many (it does not identify specific evidence)."""
    report, problems = [], []
    for q in dataset.questions:
        for i, label in enumerate(q.evidence):
            hits = [c for c in chunks if matches_label(c, label, dataset.papers)]
            report.append(
                LabelCheck(
                    question_id=q.id,
                    label_index=i,
                    paper=label.paper,
                    contains=label.contains,
                    matching_chunks=len(hits),
                    pages=sorted({int(c.get("page_start") or 0) for c in hits}),
                )
            )
            if not hits:
                problems.append(f"{q.id}[{i}] matches no chunk: {label.contains}")
            elif len(hits) > max_matches:
                problems.append(f"{q.id}[{i}] matches {len(hits)} chunks (too generic)")
    if problems:
        raise ValueError(
            "Dataset labels do not fit the indexed collection:\n" + "\n".join(problems)
        )
    return report
