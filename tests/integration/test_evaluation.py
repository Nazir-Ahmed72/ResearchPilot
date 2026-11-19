"""Evaluation runner and API end to end on the in-memory corpus with fake models."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.resources import Resources
from app.database.models import EvaluationRun, JobStatus
from app.evaluation import dataset as dataset_module
from app.evaluation.runner import EvaluationOptions, run_evaluation
from app.main import create_app
from app.workers.dispatch import InlineDispatcher
from tests.fakes import KeywordReranker, ScriptedStep
from tests.integration.test_research_graph import SIMPLE, SUFFICIENT, all_supported, good_synthesis

DATASET = {
    "name": "mini",
    "description": "test",
    "papers": {"one": {"title_contains": "Paper 1"}, "two": {"title_contains": "Paper 2"}},
    "questions": [
        {
            "id": "m1",
            "category": "simple_factual",
            "question": "What does GraphRAG build?",
            "reference_answer": "A knowledge graph of entities.",
            "evidence": [{"paper": "two", "contains": ["knowledge graph"]}],
        },
        {
            "id": "m2",
            "category": "comparison",
            "question": "Compare DPR and GraphRAG limitations",
            "reference_answer": "x",
            "evidence": [
                {"paper": "one", "contains": ["DPR"]},
                {"paper": "two", "contains": ["graph index is expensive"]},
            ],
        },
        {
            "id": "m3",
            "category": "insufficient_evidence",
            "question": "What is MMLU?",
            "answerable": False,
        },
    ],
}


@pytest.fixture
def eval_resources(resources: Resources, corpus, tmp_path: Path, monkeypatch) -> Resources:  # type: ignore[no-untyped-def]
    client, embedder, _, _ = corpus
    resources.qdrant = client
    resources.dense_embedder = embedder
    resources.reranker = KeywordReranker()
    resources.settings.qdrant_collection = "chunks"
    resources._llm_steps = {
        "analyzer": ScriptedStep(SIMPLE),
        "evaluator": ScriptedStep(SUFFICIENT),
        "synthesizer": ScriptedStep(good_synthesis),
        "verifier": ScriptedStep(all_supported),
    }
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "mini.json").write_text(json.dumps(DATASET), encoding="utf-8")
    monkeypatch.setattr(dataset_module, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr("app.evaluation.runner.RESULTS_ROOT", tmp_path / "results")
    return resources


def test_retrieval_and_generation_evaluation_end_to_end(
    eval_resources: Resources, tmp_path: Path
) -> None:
    out_dir, summary = run_evaluation(
        eval_resources,
        EvaluationOptions(
            dataset="mini",
            experiments=["E1", "E2", "E3", "E4"],
            top_k=3,
            include_generation=True,
            name="test",
        ),
        results_root=tmp_path / "results",
    )
    for name in (
        "config.json",
        "label_check.json",
        "retrieval.jsonl",
        "generation.jsonl",
        "summary.json",
        "report.md",
    ):
        assert (out_dir / name).exists(), name

    retrieval = summary["retrieval"]
    assert set(retrieval) == {"E1", "E2", "E3", "E4"}
    assert all(r["questions"] == 2 for r in retrieval.values())  # unanswerable excluded
    assert all(0 <= r["recall@3"] <= 1 for r in retrieval.values())
    sparse_rows = [
        json.loads(line)
        for line in (out_dir / "retrieval.jsonl").open()
        if json.loads(line)["experiment"] == "E2"
    ]
    m1 = next(r for r in sparse_rows if r["question_id"] == "m1")
    assert m1["recall@3"] == 1.0 and m1["retrieved"][0]["relevant_to"] == [0]

    gen = summary["generation"]
    assert gen["completed"] == 3 and gen["completion_rate"] == 1.0
    assert gen["abstention_accuracy"] is not None
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Retrieval (E1-E5)" in report and "End-to-end research workflow" in report


def test_labels_that_do_not_fit_the_collection_fail_fast(
    eval_resources: Resources, tmp_path: Path
) -> None:
    bad = {
        **DATASET,
        "name": "bad",
        "questions": [
            {
                "id": "b1",
                "category": "simple_factual",
                "question": "?",
                "evidence": [{"paper": "one", "contains": ["text that is not in any chunk"]}],
            }
        ],
    }
    (tmp_path / "datasets" / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="matches no chunk"):
        run_evaluation(
            eval_resources,
            EvaluationOptions(dataset="bad", experiments=["E1"]),
            results_root=tmp_path / "results",
        )


@pytest.fixture
def api(eval_resources: Resources) -> Iterator[tuple[TestClient, Resources]]:
    from app.services.evaluations import execute_batch

    eval_resources._dispatcher = InlineDispatcher(
        eval_resources.ingestion_pipeline,
        evaluation_runner=lambda b: execute_batch(eval_resources, b),
    )
    with TestClient(create_app(resources=eval_resources)) as client:
        yield client, eval_resources


def test_evaluation_api_runs_batch_and_lists_results(api) -> None:  # type: ignore[no-untyped-def]
    client, res = api
    resp = client.post(
        "/api/v1/evaluations/run",
        json={
            "name": "api-test",
            "dataset": "mini",
            "experiments": ["E2", "E3"],
            "top_k": 3,
            "include_generation": True,
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert [r["experiment"] for r in body["runs"]] == ["E2", "E3", "GEN"]
    for future in res.dispatcher.futures.values():  # type: ignore[attr-defined]
        future.result(timeout=60)

    items = client.get("/api/v1/evaluations/results").json()["items"]
    assert {i["experiment"] for i in items} == {"E2", "E3", "GEN"}
    assert all(i["status"] == "succeeded" and i["results_path"] for i in items)
    e2 = next(i for i in items if i["experiment"] == "E2")
    assert "recall@3" in e2["metrics"]
    assert client.get(f"/api/v1/evaluations/results/{e2['id']}").json()["experiment"] == "E2"
    only_gen = client.get("/api/v1/evaluations/results", params={"experiment": "GEN"}).json()
    assert len(only_gen["items"]) == 1


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"experiments": ["E9"]}, 400),
        ({"experiments": [], "include_generation": False}, 400),
        ({"dataset": "does_not_exist"}, 404),
        ({"name": "bad name with spaces"}, 422),
        ({"top_k": 0}, 422),
    ],
)
def test_evaluation_api_validation(api, body: dict, status: int) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    assert client.post("/api/v1/evaluations/run", json=body).status_code == status


def test_failed_batch_is_recorded(api, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, res = api

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr("app.evaluation.retrieval_eval.load_corpus", boom)
    client.post("/api/v1/evaluations/run", json={"dataset": "mini", "experiments": ["E1"]})
    for future in res.dispatcher.futures.values():  # type: ignore[attr-defined]
        future.result(timeout=60)
    with res.session_factory() as s:
        run = s.query(EvaluationRun).one()
        assert run.status is JobStatus.FAILED and "qdrant exploded" in run.error_message


def test_ragas_can_score_saved_answers_without_rerunning(
    eval_resources: Resources, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from app.evaluation import ragas_eval
    from app.evaluation.runner import score_saved_generation

    src = tmp_path / "results" / "src"
    src.mkdir(parents=True)
    rows = [
        {
            "question_id": "m1",
            "category": "simple_factual",
            "answerable": True,
            "completed": True,
            "question": "q",
            "answer": "a",
            "contexts": ["c"],
            "reference_answer": "r",
        },
        {"question_id": "m2", "category": "comparison", "answerable": True, "completed": False},
    ]
    (src / "generation.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    seen: list[list[str]] = []

    def fake_score(rows, settings, embedder, names, pace_seconds=0.0):  # type: ignore[no-untyped-def]
        seen.append([r["question_id"] for r in rows])
        for r in rows:
            r["ragas"] = {"faithfulness": 0.75, "answer_relevancy": None}

    monkeypatch.setattr(ragas_eval, "score_rows", fake_score)
    out_dir, summary = score_saved_generation(
        eval_resources,
        src,
        metrics=["faithfulness", "answer_relevancy"],
        name="r",
        results_root=tmp_path / "results",
    )
    assert seen == [["m1"]]  # only completed rows are judged
    assert summary["mean"]["faithfulness"] == 0.75 and summary["scored"]["answer_relevancy"] == 0
    assert "| m1 | simple_factual | 0.750 | – |" in (out_dir / "report.md").read_text("utf-8")
