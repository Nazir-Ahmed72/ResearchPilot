"""Retrieval engine: runs one of the five experiment strategies end to end.

| Strategy          | Queries searched           | Retrieval            | Rerank |
|-------------------|----------------------------|----------------------|--------|
| E1 dense          | original                   | dense                | no     |
| E2 sparse         | original                   | BM25                 | no     |
| E3 hybrid         | original                   | dense + BM25, RRF    | no     |
| E4 hybrid_rerank  | original                   | dense + BM25, RRF    | yes    |
| E5 hybrid_qt_rerank | original + LLM variants | dense + BM25 per variant, RRF | yes |

Reranking always scores against the *original* question: variants widen recall, but relevance
is judged against what the user actually asked.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.types import RetrievalStrategy
from app.retrieval.query_transformer import QueryTransformer
from app.retrieval.reranker import Reranker, rerank
from app.retrieval.retrievers import DenseRetriever, HybridRetriever, SparseRetriever
from app.retrieval.types import RetrievalFilters, RetrievalResult, RetrievedChunk, TransformedQuery

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    top_k: int = 8
    candidate_k: int = 30
    rerank_candidate_k: int = 20


class RerankerUnavailableError(RuntimeError):
    pass


class RetrievalEngine:
    def __init__(
        self,
        *,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        hybrid: HybridRetriever,
        reranker: Reranker | None,
        transformer: QueryTransformer | None,
        config: RetrievalConfig,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.hybrid = hybrid
        self.reranker = reranker
        self.transformer = transformer
        self.config = config

    def retrieve(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        transformed: TransformedQuery | None = None,
    ) -> RetrievalResult:
        """``transformed`` lets a caller (the LangGraph planner) supply its own query variants."""
        top_k = top_k or self.config.top_k
        timings: dict[str, float] = {}
        start = time.perf_counter()
        queries = [query]
        transformation: TransformedQuery | None = transformed

        if strategy == RetrievalStrategy.HYBRID_QT_RERANK and transformation is None:
            if self.transformer is None:
                transformation = TransformedQuery(query, [query], error="transformer unavailable")
            else:
                t = time.perf_counter()
                transformation = self.transformer.transform(query)
                timings["transform"] = _ms(t)
        if transformation is not None:
            queries = transformation.queries or [query]

        needs_rerank = strategy in (
            RetrievalStrategy.HYBRID_RERANK,
            RetrievalStrategy.HYBRID_QT_RERANK,
        )
        if needs_rerank and self.reranker is None:
            raise RerankerUnavailableError(f"Strategy {strategy.value} requires a reranker")
        fetch_k = max(top_k, self.config.rerank_candidate_k) if needs_rerank else top_k

        t = time.perf_counter()
        candidates: list[RetrievedChunk]
        if strategy == RetrievalStrategy.DENSE:
            candidates = self.dense.search(query, fetch_k, filters)
        elif strategy == RetrievalStrategy.SPARSE:
            candidates = self.sparse.search(query, fetch_k, filters)
        else:
            candidates = self.hybrid.search(queries, max(self.config.candidate_k, fetch_k), filters)
        timings["retrieve"] = _ms(t)
        candidate_count = len(candidates)

        if needs_rerank and candidates:
            assert self.reranker is not None
            t = time.perf_counter()
            chunks = rerank(self.reranker, query, candidates[:fetch_k], top_k)
            timings["rerank"] = _ms(t)
        else:
            chunks = candidates[:top_k]

        timings["total"] = _ms(start)
        logger.info(
            "Retrieval completed",
            extra={
                "strategy": strategy.value,
                "queries": len(queries),
                "candidates": candidate_count,
                "returned": len(chunks),
                "chunk_ids": [c.chunk_id for c in chunks],
                **{f"ms_{k}": v for k, v in timings.items()},
            },
        )
        return RetrievalResult(
            query=query,
            strategy=strategy,
            chunks=chunks,
            queries=queries,
            candidate_count=candidate_count,
            timings_ms=timings,
            transformation=transformation,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
