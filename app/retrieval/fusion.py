"""Reciprocal Rank Fusion (Cormack et al., 2009).

    rrf(d) = sum over lists L containing d of  weight_L / (k + rank_L(d))

Why RRF: dense cosine similarities (~0.3-0.9) and BM25 scores (0 to tens) are on incomparable
scales, and min-max normalising each list is fragile (one outlier squashes the rest). RRF uses
only ranks, needs no tuning beyond ``k`` (60 is the standard value), and rewards chunks that
several retrievers or query variants agree on.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")
    fused: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else weights[list_index]
        seen_in_list: set[str] = set()
        for rank, chunk in enumerate(ranked, start=1):
            if chunk.chunk_id in seen_in_list:  # a list should not vote twice for one chunk
                continue
            seen_in_list.add(chunk.chunk_id)
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            prev = best.get(chunk.chunk_id)
            # Keep the per-retriever scores/ranks from every list (best score wins per stage).
            if prev is None:
                best[chunk.chunk_id] = chunk
            else:
                merged_scores = dict(prev.scores)
                merged_ranks = dict(prev.ranks)
                for stage, value in chunk.scores.items():
                    if value > merged_scores.get(stage, float("-inf")):
                        merged_scores[stage] = value
                for stage, value in chunk.ranks.items():
                    merged_ranks[stage] = min(value, merged_ranks.get(stage, value))
                best[chunk.chunk_id] = RetrievedChunk(
                    prev.chunk_id, prev.payload, prev.score, merged_scores, merged_ranks
                )
    # Ties broken by chunk_id for deterministic output.
    ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return [best[cid].with_stage("rrf", fused[cid], rank) for rank, cid in enumerate(ordered, 1)]
