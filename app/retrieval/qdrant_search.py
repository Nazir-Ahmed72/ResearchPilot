"""Low-level Qdrant search shared by the dense, sparse and hybrid retrievers.

All searches needed for one request (e.g. dense + sparse for each of 4 query variants) are sent
in a single ``query_batch_points`` call, so a remote cluster costs one network round trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient, models

from app.embeddings.sparse import SparseEmbedding
from app.retrieval.types import RetrievalFilters, RetrievedChunk
from app.vectorstore.qdrant_store import DENSE_VECTOR, SPARSE_VECTOR

SearchKind = Literal["dense", "sparse"]


@dataclass(frozen=True)
class SearchRequest:
    kind: SearchKind
    vector: Sequence[float] | SparseEmbedding
    limit: int


def build_filter(filters: RetrievalFilters | None) -> models.Filter | None:
    if filters is None or filters.is_empty:
        return None
    must: list[models.Condition] = []
    if filters.collection_id:
        must.append(
            models.FieldCondition(
                key="collection_id", match=models.MatchValue(value=filters.collection_id)
            )
        )
    if filters.paper_ids:
        must.append(
            models.FieldCondition(
                key="paper_id", match=models.MatchAny(any=list(filters.paper_ids))
            )
        )
    if filters.sections:
        must.append(
            models.FieldCondition(key="section", match=models.MatchAny(any=list(filters.sections)))
        )
    if filters.year_from is not None or filters.year_to is not None:
        must.append(
            models.FieldCondition(
                key="year", range=models.Range(gte=filters.year_from, lte=filters.year_to)
            )
        )
    return models.Filter(must=must)


class QdrantSearcher:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._exists = False

    def _collection_exists(self) -> bool:
        if not self._exists:  # cache only the positive answer; it never becomes false normally
            self._exists = self.client.collection_exists(self.collection)
        return self._exists

    def search(
        self, requests: Sequence[SearchRequest], filters: RetrievalFilters | None = None
    ) -> list[list[RetrievedChunk]]:
        if not requests:
            return []
        if not self._collection_exists():
            return [[] for _ in requests]  # nothing has been ingested yet
        query_filter = build_filter(filters)
        batch: list[models.QueryRequest] = []
        for req in requests:
            if req.kind == "dense":
                query: object = list(req.vector)  # type: ignore[arg-type]
                using = DENSE_VECTOR
            else:
                sparse = req.vector
                assert isinstance(sparse, SparseEmbedding)
                query = models.SparseVector(indices=sparse.indices, values=sparse.values)
                using = SPARSE_VECTOR
            batch.append(
                models.QueryRequest(
                    query=query,
                    using=using,
                    filter=query_filter,
                    limit=req.limit,
                    with_payload=True,
                )
            )
        responses = self.client.query_batch_points(self.collection, requests=batch)
        results: list[list[RetrievedChunk]] = []
        for req, response in zip(requests, responses, strict=True):
            results.append(
                [
                    RetrievedChunk(
                        chunk_id=str(point.id),
                        payload=point.payload or {},
                        score=float(point.score),
                        scores={req.kind: float(point.score)},
                        ranks={req.kind: rank},
                    )
                    for rank, point in enumerate(response.points, start=1)
                ]
            )
        return results
