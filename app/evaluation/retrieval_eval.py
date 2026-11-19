"""Retrieval experiments E1-E5 against the labelled dataset.

Retrieval is evaluated separately from generation: each strategy returns its top-K chunks for
every answerable question and is scored with Recall@K, Precision@K, MRR and nDCG@K, plus
latency. Unanswerable questions have no relevant chunks and are excluded here (they are scored
in the generation evaluation).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client import QdrantClient

from app.core.types import RetrievalStrategy
from app.evaluation.dataset import EvalDataset, matches_label, relevance
from app.evaluation.metrics import mean, percentile, retrieval_metrics
from app.retrieval.engine import RetrievalEngine
from app.retrieval.qdrant_search import build_filter
from app.retrieval.types import RetrievalFilters

logger = logging.getLogger(__name__)

EXPERIMENTS: dict[str, RetrievalStrategy] = {
    "E1": RetrievalStrategy.DENSE,
    "E2": RetrievalStrategy.SPARSE,
    "E3": RetrievalStrategy.HYBRID,
    "E4": RetrievalStrategy.HYBRID_RERANK,
    "E5": RetrievalStrategy.HYBRID_QT_RERANK,
}


def load_corpus(
    client: QdrantClient, collection: str, filters: RetrievalFilters | None = None
) -> list[dict[str, Any]]:
    """All chunk payloads in scope (used to count how many relevant chunks exist)."""
    if not client.collection_exists(collection):
        return []
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            scroll_filter=build_filter(filters),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(p.payload or {} for p in points)
        if offset is None:
            return payloads


def relevant_counts(dataset: EvalDataset, corpus: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        q.id: sum(
            1 for c in corpus if any(matches_label(c, lab, dataset.papers) for lab in q.evidence)
        )
        for q in dataset.questions
    }


def evaluate_retrieval(
    engine: RetrievalEngine,
    dataset: EvalDataset,
    *,
    experiments: Sequence[str],
    k: int,
    corpus: Sequence[Mapping[str, Any]],
    filters: RetrievalFilters | None = None,
) -> list[dict[str, Any]]:
    counts = relevant_counts(dataset, corpus)
    rows: list[dict[str, Any]] = []
    answerable = [q for q in dataset.questions if q.answerable]
    for exp in experiments:
        strategy = EXPERIMENTS[exp]
        for q in answerable:
            start = time.perf_counter()
            try:
                result = engine.retrieve(q.question, strategy=strategy, top_k=k, filters=filters)
            except Exception as exc:  # noqa: BLE001 - one failure must not abort the experiment
                logger.warning(
                    "Retrieval failed", extra={"experiment": exp, "question": q.id}, exc_info=True
                )
                rows.append(
                    {
                        "experiment": exp,
                        "strategy": strategy.value,
                        "question_id": q.id,
                        "category": q.category,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            payloads = [c.payload for c in result.chunks]
            rel = relevance(payloads, q, dataset.papers)
            rows.append(
                {
                    "experiment": exp,
                    "strategy": strategy.value,
                    "question_id": q.id,
                    "category": q.category,
                    "queries": result.queries,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                    "timings_ms": result.timings_ms,
                    "retrieved": [
                        {
                            "rank": i,
                            "chunk_id": c.chunk_id,
                            "title": c.title,
                            "section": c.section,
                            "page_start": c.payload.get("page_start"),
                            "relevant_to": sorted(r),
                        }
                        for i, (c, r) in enumerate(zip(result.chunks, rel, strict=True), start=1)
                    ],
                    "n_labels": len(q.evidence),
                    "n_relevant_in_corpus": counts[q.id],
                    **retrieval_metrics(rel, len(q.evidence), counts[q.id], k),
                }
            )
    return rows


def summarize_retrieval(rows: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    metric_keys = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
    by_exp: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_exp[r["experiment"]].append(r)
    summary: dict[str, Any] = {}
    for exp, items in by_exp.items():
        ok = [r for r in items if "error" not in r]
        latencies = [r["latency_ms"] for r in ok]
        categories: dict[str, dict[str, float | None]] = {}
        for cat in sorted({r["category"] for r in ok}):
            cat_rows = [r for r in ok if r["category"] == cat]
            categories[cat] = {m: mean(r[m] for r in cat_rows) for m in metric_keys}
        summary[exp] = {
            "strategy": items[0]["strategy"],
            "questions": len(items),
            "errors": len(items) - len(ok),
            **{m: mean(r[m] for r in ok) for m in metric_keys},
            "latency_ms_mean": mean(latencies),
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p95": percentile(latencies, 0.95),
            "by_category": categories,
        }
    return summary
