"""Qdrant collection management and chunk upsert/delete.

One collection holds every chunk, with two named vectors per point:
* ``dense``: BGE embedding, cosine distance
* ``bm25``:  sparse BM25 term weights, with Qdrant's IDF modifier

Papers and collections are separated by payload filters (``paper_id``, ``collection_id``)
backed by payload indexes; Qdrant Cloud's strict mode requires an index for filtered fields.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.embeddings.sparse import SparseEmbedding

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
KEYWORD_INDEXES = ("paper_id", "collection_id", "section")
INTEGER_INDEXES = ("year", "page_start")


class VectorStoreConfigError(RuntimeError):
    """The existing collection is incompatible with the configured embedding model."""


@dataclass(frozen=True)
class ChunkPoint:
    id: str
    dense: Sequence[float]
    sparse: SparseEmbedding
    payload: dict[str, Any]


class QdrantChunkStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._ready_dimension: int | None = None

    def ensure_collection(self, dense_dimension: int) -> None:
        """Create the collection and payload indexes if missing; verify the dimension if not."""
        if self._ready_dimension == dense_dimension:
            return
        if self.client.collection_exists(self.collection):
            existing = self.client.get_collection(self.collection).config.params.vectors
            size = existing[DENSE_VECTOR].size if isinstance(existing, dict) else None
            if size != dense_dimension:
                raise VectorStoreConfigError(
                    f"Collection {self.collection!r} has dense size {size}, but the embedding "
                    f"model produces {dense_dimension}. Use a new QDRANT_COLLECTION or re-index."
                )
        else:
            logger.info("Creating Qdrant collection", extra={"collection": self.collection})
            self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dense_dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        for field in KEYWORD_INDEXES:
            self._index(field, models.PayloadSchemaType.KEYWORD)
        for field in INTEGER_INDEXES:
            self._index(field, models.PayloadSchemaType.INTEGER)
        self._ready_dimension = dense_dimension

    def _index(self, field: str, schema: models.PayloadSchemaType) -> None:
        # Idempotent on Qdrant servers; a no-op in embedded local mode, which warns about it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            self.client.create_payload_index(self.collection, field, field_schema=schema, wait=True)

    def upsert(self, points: Iterable[ChunkPoint], batch_size: int = 64) -> int:
        batch: list[models.PointStruct] = []
        total = 0
        for p in points:
            batch.append(
                models.PointStruct(
                    id=p.id,
                    vector={
                        DENSE_VECTOR: list(p.dense),
                        SPARSE_VECTOR: models.SparseVector(
                            indices=p.sparse.indices, values=p.sparse.values
                        ),
                    },
                    payload=p.payload,
                )
            )
            if len(batch) >= batch_size:
                total += self._flush(batch)
        total += self._flush(batch)
        return total

    def _flush(self, batch: list[models.PointStruct]) -> int:
        if not batch:
            return 0
        self.client.upsert(self.collection, points=batch, wait=True)
        n = len(batch)
        batch.clear()
        return n

    @staticmethod
    def paper_filter(paper_id: UUID | str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="paper_id", match=models.MatchValue(value=str(paper_id)))
            ]
        )

    def delete_paper(self, paper_id: UUID | str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self.paper_filter(paper_id)),
            wait=True,
        )

    def count(self, paper_id: UUID | str | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        flt = self.paper_filter(paper_id) if paper_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count
