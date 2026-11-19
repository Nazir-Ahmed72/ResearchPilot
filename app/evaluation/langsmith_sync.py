"""Publish evaluation results to LangSmith as a dataset plus one experiment per strategy.

The dataset (questions, reference answers, evidence labels) is created once per name. Each
experiment's "target" does NOT re-run retrieval: it returns the results already computed by the
local evaluation, keyed by question id. So publishing costs no extra Qdrant or LLM calls, and
LangSmith shows exactly the numbers in the local report, side by side per experiment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.core.config import Settings
from app.evaluation.dataset import EvalDataset
from app.observability.tracing import langsmith_client

logger = logging.getLogger(__name__)


def ensure_dataset(client: Any, dataset: EvalDataset) -> str:
    name = f"researchpilot-{dataset.name}"
    if client.has_dataset(dataset_name=name):
        return name
    client.create_dataset(dataset_name=name, description=dataset.description)
    client.create_examples(
        dataset_name=name,
        examples=[
            {
                "inputs": {"question_id": q.id, "question": q.question},
                "outputs": {
                    "reference_answer": q.reference_answer,
                    "answerable": q.answerable,
                    "evidence": [e.model_dump() for e in q.evidence],
                },
                "metadata": {"category": q.category},
            }
            for q in dataset.questions
        ],
    )
    return name


def _metric_evaluator(key: str):  # type: ignore[no-untyped-def]
    def evaluator(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": key, "score": outputs.get(key)}

    evaluator.__name__ = key.replace("@", "_at_")
    return evaluator


def sync_to_langsmith(
    settings: Settings,
    dataset: EvalDataset,
    retrieval_rows: Sequence[Mapping[str, Any]],
    k: int,
) -> dict[str, Any]:
    client = langsmith_client(settings)
    dataset_name = ensure_dataset(client, dataset)
    metric_keys = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", "latency_ms"]
    experiments: dict[str, str] = {}
    for exp in sorted({r["experiment"] for r in retrieval_rows}):
        rows = {
            r["question_id"]: r
            for r in retrieval_rows
            if r["experiment"] == exp and "error" not in r
        }
        strategy = next(iter(rows.values()))["strategy"] if rows else exp

        def target(inputs: dict[str, Any], _rows: Mapping[str, Any] = rows) -> dict[str, Any]:
            row = _rows.get(inputs["question_id"])
            if row is None:  # unanswerable questions are not part of retrieval scoring
                return {"retrieved": [], **{m: None for m in metric_keys}}
            return {
                "retrieved": row["retrieved"],
                "queries": row["queries"],
                **{m: row.get(m) for m in metric_keys},
            }

        result = client.evaluate(
            target,
            data=dataset_name,
            evaluators=[_metric_evaluator(m) for m in metric_keys],
            experiment_prefix=f"{exp}-{strategy}",
            metadata={
                "experiment": exp,
                "strategy": strategy,
                "top_k": k,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "llm_fast_model": settings.llm_fast_model,
            },
            max_concurrency=1,
        )
        experiments[exp] = getattr(result, "experiment_name", str(result))
        logger.info("Published experiment to LangSmith", extra={"experiment": exp})
    return {"dataset": dataset_name, "experiments": experiments}
