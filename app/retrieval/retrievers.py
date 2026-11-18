"""Dense, sparse and hybrid retrievers. Each is independently usable and testable.

A sparse query made only of stop words has no terms and is skipped rather than sent.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.embeddings.dense import DenseEmbedder
from app.embeddings.sparse import Bm25SparseEncoder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.qdrant_search import QdrantSearcher, SearchRequest
from app.retrieval.types import RetrievalFilters, RetrievedChunk


class DenseRetriever:
    def __init__(self, searcher: QdrantSearcher, embedder: DenseEmbedder) -> None:
        self.searcher = searcher
        self.embedder = embedder

    def search(
        self, query: str, top_k: int, filters: RetrievalFilters | None = None
    ) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        return self.searcher.search([SearchRequest("dense", vector, top_k)], filters)[0]


class SparseRetriever:
    def __init__(self, searcher: QdrantSearcher, encoder: Bm25SparseEncoder) -> None:
        self.searcher = searcher
        self.encoder = encoder

    def search(
        self, query: str, top_k: int, filters: RetrievalFilters | None = None
    ) -> list[RetrievedChunk]:
        vector = self.encoder.embed_query(query)
        if not vector.indices:
            return []
        return self.searcher.search([SearchRequest("sparse", vector, top_k)], filters)[0]


class HybridRetriever:
    """Dense + sparse for every query variant, all in one Qdrant round trip, fused with RRF."""

    def __init__(
        self,
        searcher: QdrantSearcher,
        embedder: DenseEmbedder,
        encoder: Bm25SparseEncoder,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self.searcher = searcher
        self.embedder = embedder
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(
        self,
        queries: Sequence[str],
        candidate_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        requests: list[SearchRequest] = []
        weights: list[float] = []
        for query in queries:
            requests.append(SearchRequest("dense", self.embedder.embed_query(query), candidate_k))
            weights.append(self.dense_weight)
            sparse = self.encoder.embed_query(query)
            if sparse.indices:
                requests.append(SearchRequest("sparse", sparse, candidate_k))
                weights.append(self.sparse_weight)
        ranked_lists = self.searcher.search(requests, filters)
        return reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)
