"""Evaluation building blocks: metrics, dataset labels, scoring, report, tracing, Ragas inputs."""

from __future__ import annotations

import math

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.evaluation import metrics as m
from app.evaluation.dataset import (
    EvalDataset,
    EvalQuestion,
    load_dataset,
    matches_label,
    relevance,
    validate,
)
from app.evaluation.generation_eval import answer_text, score_question, summarize_generation
from app.evaluation.report import render_report

# --- metrics (hand-computed) ------------------------------------------------------------------

REL = [set(), {0}, set(), {0, 1}, set()]  # ranks 2 and 4 relevant; rank 4 covers both labels


def test_recall_counts_labels_not_chunks() -> None:
    assert m.recall_at_k(REL, n_labels=2, k=2) == 0.5
    assert m.recall_at_k(REL, n_labels=2, k=5) == 1.0
    assert m.recall_at_k([], n_labels=2, k=5) == 0.0


def test_precision_mrr_ndcg() -> None:
    assert m.precision_at_k(REL, 5) == 0.4
    assert m.reciprocal_rank(REL, 5) == 0.5
    assert m.reciprocal_rank(REL, 1) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m.ndcg_at_k(REL, n_relevant=2, k=5) == pytest.approx(dcg / idcg)
    assert m.ndcg_at_k([{0}, {0}], n_relevant=2, k=2) == pytest.approx(1.0)
    assert m.ndcg_at_k([set()], n_relevant=0, k=5) == 0.0


def test_mean_and_percentile_ignore_none() -> None:
    assert m.mean([1.0, None, 0.0]) == 0.5 and m.mean([None]) is None
    assert m.percentile([10, 20, 30, 40], 0.5) == 20 and m.percentile([], 0.5) is None


# --- dataset -------------------------------------------------------------------------------------

PAPERS = {"a": {"title_contains": "Paper One"}}


def ds(*questions: dict) -> EvalDataset:  # type: ignore[type-arg]
    return EvalDataset.model_validate({"name": "t", "papers": PAPERS, "questions": list(questions)})


def chunk(text: str, title: str = "Paper One: a study", page: int = 1) -> dict:  # type: ignore[type-arg]
    return {"title": title, "text": text, "page_start": page}


def test_bundled_dataset_is_well_formed() -> None:
    data = load_dataset()
    cats = {q.category for q in data.questions}
    assert cats == {
        "simple_factual",
        "comparison",
        "multi_document",
        "limitations",
        "multi_query",
        "insufficient_evidence",
    }
    assert sum(not q.answerable for q in data.questions) >= 3


def test_label_matching_is_case_whitespace_insensitive_with_alternatives() -> None:
    data = ds(
        {
            "id": "q",
            "category": "simple_factual",
            "question": "?",
            "evidence": [{"paper": "a", "contains": ["Dense  Passage", "BM25|lucene"]}],
        }
    )
    label, papers = data.questions[0].evidence[0], data.papers
    assert matches_label(chunk("we use dense passage retrieval vs Lucene"), label, papers)
    assert not matches_label(chunk("dense passage only"), label, papers)
    assert not matches_label(chunk("dense passage BM25", title="Other paper"), label, papers)


def test_relevance_maps_chunks_to_labels() -> None:
    data = ds(
        {
            "id": "q",
            "category": "comparison",
            "question": "?",
            "evidence": [
                {"paper": "a", "contains": ["alpha"]},
                {"paper": "a", "contains": ["beta"]},
            ],
        }
    )
    rel = relevance(
        [chunk("alpha"), chunk("x"), chunk("alpha beta")], data.questions[0], data.papers
    )
    assert rel == [{0}, set(), {0, 1}]


def test_validate_rejects_unmatched_and_generic_labels() -> None:
    data = ds(
        {
            "id": "q1",
            "category": "simple_factual",
            "question": "?",
            "evidence": [{"paper": "a", "contains": ["missing phrase"]}],
        }
    )
    with pytest.raises(ValueError, match="matches no chunk"):
        validate(data, [chunk("something else")])
    generic = ds(
        {
            "id": "q2",
            "category": "simple_factual",
            "question": "?",
            "evidence": [{"paper": "a", "contains": ["the"]}],
        }
    )
    with pytest.raises(ValueError, match="too generic"):
        validate(generic, [chunk("the x")] * 5)


@pytest.mark.parametrize(
    "question",
    [
        {"id": "q", "category": "simple_factual", "question": "?", "evidence": []},
        {
            "id": "q",
            "category": "insufficient_evidence",
            "question": "?",
            "answerable": False,
            "evidence": [{"paper": "a", "contains": ["x"]}],
        },
        {
            "id": "q",
            "category": "simple_factual",
            "question": "?",
            "evidence": [{"paper": "zzz", "contains": ["x"]}],
        },
    ],
)
def test_inconsistent_questions_rejected(question: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValidationError):
        ds(question)


# --- generation scoring ----------------------------------------------------------------------


def _values(text: str, sufficient: bool = True) -> dict:  # type: ignore[type-arg]
    ev = {"chunk_id": "c1", "title": "Paper One: a study", "text": text, "eid": "E1"}
    return {
        "evidence": [ev],
        "final": {
            "evidence_sufficient": sufficient,
            "references": [{"chunk_id": "c1"}],
            "summary": "S",
            "sections": [{"claims": [{"text": "claim A"}]}],
            "verification": {"citation_validity": 1.0},
        },
    }


def test_score_question_label_metrics() -> None:
    data = ds(
        {
            "id": "q",
            "category": "simple_factual",
            "question": "?",
            "evidence": [{"paper": "a", "contains": ["alpha"]}],
        }
    )
    good = score_question(data.questions[0], _values("alpha here"), data)
    bad = score_question(data.questions[0], _values("unrelated", sufficient=False), data)
    assert good["citation_label_accuracy"] == 1.0 and good["evidence_recall"] == 1.0
    assert good["false_abstention"] is False and good["abstained_correctly"] is None
    assert bad["citation_label_accuracy"] == 0.0 and bad["false_abstention"] is True


def test_unanswerable_question_scores_abstention() -> None:
    data = ds(
        {"id": "q", "category": "insufficient_evidence", "question": "?", "answerable": False}
    )
    assert (
        score_question(data.questions[0], _values("x", sufficient=False), data)[
            "abstained_correctly"
        ]
        is True
    )
    assert (
        score_question(data.questions[0], _values("x", sufficient=True), data)[
            "abstained_correctly"
        ]
        is False
    )


def test_answer_text_joins_summary_and_claims() -> None:
    assert answer_text({"summary": "S", "sections": [{"claims": [{"text": "A"}]}]}) == "S\nA"


def test_summarize_generation() -> None:
    rows = [
        {
            "completed": True,
            "answerable": True,
            "latency_ms": 100.0,
            "false_abstention": False,
            "citation_label_accuracy": 1.0,
            "evidence_recall": 0.5,
            "retry_count": 1,
            "token_usage": {
                "m": {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}
            },
            "node_timings_ms": {"synthesize": 50.0},
            "ragas": {"faithfulness": 0.8},
        },
        {
            "completed": True,
            "answerable": False,
            "latency_ms": 300.0,
            "abstained_correctly": True,
            "token_usage": {},
            "node_timings_ms": {},
        },
        {"completed": False, "answerable": True, "latency_ms": 5.0},
    ]
    s = summarize_generation(rows, pricing={"m": {"input": 2.0, "output": 0.0}})
    assert s["completion_rate"] == round(2 / 3, 4) and s["abstention_accuracy"] == 1.0
    assert s["retry_rate"] == 0.5 and s["ragas"] == {"faithfulness": 0.8}
    assert s["cost_usd_per_query"] == 1.0  # $2 for 1M input tokens, averaged over 2 runs
    assert (
        summarize_generation(rows, pricing={"other": {"input": 1, "output": 1}})[
            "cost_usd_per_query"
        ]
        is None
    )


def test_report_contains_tables_and_caveats() -> None:
    meta = {
        "name": "n",
        "dataset": "d",
        "questions": 3,
        "answerable": 2,
        "papers": 1,
        "created_at": "now",
        "top_k": 5,
        "generation_strategy": "hybrid_rerank",
        "models": {"main": "a", "fast": "b", "embedding": "c", "reranker": "d"},
    }
    retrieval = {
        "E1": {
            "strategy": "dense",
            "recall@5": 0.5,
            "precision@5": 0.2,
            "mrr": 0.3,
            "ndcg@5": 0.4,
            "latency_ms_mean": 120.0,
            "latency_ms_p95": 200.0,
            "errors": 0,
            "by_category": {"comparison": {"recall@5": 0.5}},
        }
    }
    gen = summarize_generation(
        [
            {
                "completed": True,
                "answerable": True,
                "latency_ms": 1.0,
                "token_usage": {},
                "node_timings_ms": {"respond": 1.0},
            }
        ]
    )
    report = render_report(meta, retrieval, gen)
    assert "| E1 | `dense` | 0.500 |" in report and "Recall@5 by question category" in report
    assert "LLM-as-a-judge" in report and "Completion rate" in report


# --- tracing -------------------------------------------------------------------------------------


def test_tracing_is_off_by_default_and_redacts_when_on() -> None:
    from app.observability.tracing import create_tracer, redact_payload

    assert create_tracer(Settings(_env_file=None)) is None  # type: ignore[call-arg]
    tracer = create_tracer(
        Settings(
            _env_file=None,
            langsmith_tracing=True,  # type: ignore[call-arg]
            langsmith_api_key=SecretStr("lsv2_pt_testkey123456"),
        )
    )
    assert tracer is not None and tracer.project_name == "researchpilot"
    out = redact_payload({"api_key": "x", "input_tokens": 3, "q": "postgresql://u:hunter2@h/db"})
    assert out["api_key"] != "x" and out["input_tokens"] == 3 and "hunter2" not in out["q"]


# --- Ragas input mapping -------------------------------------------------------------------------


def test_ragas_inputs_skip_reference_metrics_without_reference() -> None:
    from app.evaluation.ragas_eval import _inputs

    row = {"question": "q", "answer": "a", "contexts": ["c"], "reference_answer": ""}
    assert _inputs("faithfulness", row) == {
        "user_input": "q",
        "response": "a",
        "retrieved_contexts": ["c"],
    }
    assert _inputs("answer_correctness", row) is None and _inputs("context_recall", row) is None
    assert _inputs("faithfulness", {**row, "contexts": []}) is None


def test_ragas_requires_judge_key() -> None:
    from app.evaluation.ragas_eval import build_metrics
    from tests.fakes import FakeDenseEmbedder

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_metrics(Settings(_env_file=None), FakeDenseEmbedder(), ["faithfulness"])  # type: ignore[call-arg]


def test_eval_question_model_roundtrip() -> None:
    q = EvalQuestion(
        id="x", category="limitations", question="?", evidence=[{"paper": "a", "contains": ["y"]}]
    )  # type: ignore[list-item]
    assert q.answerable and q.evidence[0].contains == ["y"]
