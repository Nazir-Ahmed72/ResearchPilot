"""Shared integration fixtures: a small two-paper corpus indexed in in-memory Qdrant."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from app.embeddings.sparse import Bm25SparseEncoder
from app.ingestion.chunker import chunk_id_for
from app.retrieval.qdrant_search import QdrantSearcher
from app.vectorstore.qdrant_store import ChunkPoint, QdrantChunkStore
from tests.corpus import CORPUS
from tests.fakes import FakeDenseEmbedder


@pytest.fixture
def corpus() -> Iterator[tuple[QdrantClient, FakeDenseEmbedder, Bm25SparseEncoder, QdrantSearcher]]:
    client = QdrantClient(location=":memory:")
    embedder, encoder = FakeDenseEmbedder(dimension=64), Bm25SparseEncoder()
    store = QdrantChunkStore(client, "chunks")
    store.ensure_collection(embedder.dimension)
    points = []
    for i, (paper, coll, section, year, text) in enumerate(CORPUS):
        cid = chunk_id_for(paper, i)
        payload = {
            "chunk_id": cid,
            "paper_id": paper,
            "collection_id": coll,
            "section": section,
            "year": year,
            "text": text,
            "title": f"Paper {paper[0]}",
            "page_start": i + 1,
            "page_end": i + 1,
        }
        points.append(
            ChunkPoint(
                cid, embedder.embed_documents([text])[0], encoder.embed_document(text), payload
            )
        )
    store.upsert(points)
    yield client, embedder, encoder, QdrantSearcher(client, "chunks")
    client.close()
