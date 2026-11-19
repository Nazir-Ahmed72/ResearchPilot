"""Retrieval metrics over content-based relevance labels.

``rel`` is, for each retrieved chunk in rank order, the set of evidence labels it satisfies
(empty set = not relevant). ``n_labels`` is the number of labelled evidence items for the
question; ``n_relevant`` the number of chunks in the whole collection that satisfy any label.

* Recall@K    = labels covered by the top K / n_labels   ("did we find each needed piece?")
* Precision@K = relevant chunks in the top K / K
* MRR         = 1 / rank of the first relevant chunk (0 if none in the top K)
* nDCG@K      = binary-gain DCG of the top K / ideal DCG with min(K, n_relevant) relevant chunks
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def recall_at_k(rel: Sequence[set[int]], n_labels: int, k: int) -> float:
    if n_labels == 0:
        return 0.0
    covered: set[int] = set().union(*rel[:k]) if rel[:k] else set()
    return len(covered) / n_labels


def precision_at_k(rel: Sequence[set[int]], k: int) -> float:
    return sum(1 for r in rel[:k] if r) / k if k else 0.0


def reciprocal_rank(rel: Sequence[set[int]], k: int) -> float:
    for rank, r in enumerate(rel[:k], start=1):
        if r:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rel: Sequence[set[int]], n_relevant: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, r in enumerate(rel[:k], start=1) if r)
    ideal_hits = min(k, n_relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(
    rel: Sequence[set[int]], n_labels: int, n_relevant: int, k: int
) -> dict[str, float]:
    return {
        f"recall@{k}": round(recall_at_k(rel, n_labels, k), 4),
        f"precision@{k}": round(precision_at_k(rel, k), 4),
        "mrr": round(reciprocal_rank(rel, k), 4),
        f"ndcg@{k}": round(ndcg_at_k(rel, n_relevant, k), 4),
    }


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    index = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
    return round(vals[index], 1)
