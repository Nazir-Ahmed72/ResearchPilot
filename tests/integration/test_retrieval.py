"""Retrievers and the engine against in-memory Qdrant, with a small hand-made corpus."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.types import RetrievalStrategy
from app.embeddings.sparse import Bm25SparseEncoder
from app.ingestion.chunker import chunk_id_for
from app.retrieval.engine import RerankerUnavailableError, RetrievalConfig, RetrievalEngine
from app.retrieval.qdrant_search import QdrantSearcher
from app.retrieval.query_transformer import QueryPlan, QueryTransformer
from app.retrieval.retrievers import DenseRetriever, HybridRetriever, SparseRetriever
from app.retrieval.types import RetrievalFilters
from app.vectorstore.qdrant_store import ChunkPoint, QdrantChunkStore
from tests.fakes import FakeDenseEmbedder, FakePlanner, KeywordReranker

PAPER_A, PAPER_B = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
COLL_X, COLL_Y = "c-x", "c-y"
CORPUS = [
    (
        PAPER_A,
        COLL_X,
        "1 Introduction",
        2020,
        "Retrieval augmented generation conditions a language model on retrieved passages.",
    ),
    (
        PAPER_A,
        COLL_X,
        "3 Method",
        2020,
        "We use DPR, a dense passage retriever with a BERT bi-encoder, and BM25 as a baseline.",
    ),
    (
        PAPER_A,
        COLL_X,
        "5 Limitations",
        2020,
        "Limitations include retrieval errors that propagate into hallucinated answers.",
    ),
    (
        PAPER_B,
        COLL_Y,
        "1 Introduction",
        2024,
        "GraphRAG builds a knowledge graph of entities and summarises graph communities.",
    ),
    (
        PAPER_B,
        COLL_Y,
        "4 Results",
        2024,
        "GraphRAG improves comprehensiveness on global sensemaking questions over naive RAG.",
    ),
    (
        PAPER_B,
        COLL_Y,
        "6 Limitations",
        2024,
        "Building the graph index is expensive, and evaluation relies on an LLM judge.",
    ),
]


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


def _texts(chunks) -> list[str]:  # type: ignore[no-untyped-def]
    return [c.text for c in chunks]


def test_sparse_finds_exact_keyword(corpus) -> None:  # type: ignore[no-untyped-def]
    _, _, encoder, searcher = corpus
    results = SparseRetriever(searcher, encoder).search("DPR", top_k=3)
    assert len(results) == 1 and "DPR" in results[0].text
    assert results[0].scores["sparse"] > 0 and results[0].ranks == {"sparse": 1}


def test_sparse_stopword_only_query_returns_nothing(corpus) -> None:  # type: ignore[no-untyped-def]
    _, _, encoder, searcher = corpus
    assert SparseRetriever(searcher, encoder).search("the of and", top_k=3) == []


def test_dense_returns_top_k_with_provenance(corpus) -> None:  # type: ignore[no-untyped-def]
    _, embedder, _, searcher = corpus
    results = DenseRetriever(searcher, embedder).search("graph communities entities", top_k=2)
    assert len(results) == 2
    assert results[0].paper_id == PAPER_B
    assert {"section", "page_start", "title", "chunk_id"} <= set(results[0].payload)


def test_hybrid_fuses_and_deduplicates(corpus) -> None:  # type: ignore[no-untyped-def]
    _, embedder, encoder, searcher = corpus
    results = HybridRetriever(searcher, embedder, encoder).search(["GraphRAG limitations"], 10)
    ids = [c.chunk_id for c in results]
    assert len(ids) == len(set(ids)) == len(CORPUS)
    top = results[0]
    assert "rrf" in top.scores and {"dense", "sparse"} <= set(top.scores)
    assert "graph" in top.text.lower()


@pytest.mark.parametrize(
    ("filters", "expected_papers"),
    [
        (RetrievalFilters(collection_id=COLL_X), {PAPER_A}),
        (RetrievalFilters(paper_ids=(PAPER_B,)), {PAPER_B}),
        (RetrievalFilters(year_from=2021), {PAPER_B}),
        (RetrievalFilters(year_to=2020), {PAPER_A}),
        (RetrievalFilters(sections=("5 Limitations", "6 Limitations")), {PAPER_A, PAPER_B}),
    ],
)
def test_metadata_filters(corpus, filters: RetrievalFilters, expected_papers: set[str]) -> None:  # type: ignore[no-untyped-def]
    _, embedder, encoder, searcher = corpus
    results = HybridRetriever(searcher, embedder, encoder).search(
        ["limitations retrieval"], 10, filters
    )
    assert results and {c.paper_id for c in results} == expected_papers


def test_missing_collection_returns_empty(corpus) -> None:  # type: ignore[no-untyped-def]
    client, embedder, _, _ = corpus
    searcher = QdrantSearcher(client, "does-not-exist")
    assert DenseRetriever(searcher, embedder).search("anything", 5) == []


def _engine(corpus, reranker=None, planner=None) -> RetrievalEngine:  # type: ignore[no-untyped-def]
    _, embedder, encoder, searcher = corpus
    return RetrievalEngine(
        dense=DenseRetriever(searcher, embedder),
        sparse=SparseRetriever(searcher, encoder),
        hybrid=HybridRetriever(searcher, embedder, encoder),
        reranker=reranker,
        transformer=QueryTransformer(planner, max_queries=4) if planner else None,
        config=RetrievalConfig(top_k=3, candidate_k=10, rerank_candidate_k=5),
    )


@pytest.mark.parametrize("strategy", list(RetrievalStrategy))
def test_every_strategy_returns_ranked_results(corpus, strategy: RetrievalStrategy) -> None:  # type: ignore[no-untyped-def]
    planner = FakePlanner(QueryPlan(complexity="simple", rewritten_query="RAG limitations"))
    engine = _engine(corpus, KeywordReranker(), planner)
    result = engine.retrieve(
        "What are the limitations of retrieval augmented generation?", strategy=strategy
    )
    assert result.strategy is strategy and 0 < len(result.chunks) <= 3
    final_stage = {
        RetrievalStrategy.DENSE: "dense",
        RetrievalStrategy.SPARSE: "sparse",
        RetrievalStrategy.HYBRID: "rrf",
    }.get(strategy, "rerank")
    assert all(final_stage in c.scores for c in result.chunks)
    assert "total" in result.timings_ms and "retrieve" in result.timings_ms
    assert ("rerank" in result.timings_ms) == (final_stage == "rerank")


def test_query_transformation_widens_search_and_reranks_against_original(corpus) -> None:  # type: ignore[no-untyped-def]
    planner = FakePlanner(
        QueryPlan(
            complexity="complex",
            rewritten_query="RAG versus GraphRAG",
            sub_questions=["How does GraphRAG build its index?", "What is DPR retrieval?"],
        )
    )
    reranker = KeywordReranker()
    result = _engine(corpus, reranker, planner).retrieve(
        "Compare RAG and GraphRAG", strategy=RetrievalStrategy.HYBRID_QT_RERANK
    )
    assert result.queries[0] == "Compare RAG and GraphRAG" and len(result.queries) == 4
    assert result.transformation is not None and result.transformation.used_llm
    assert "transform" in result.timings_ms and reranker.calls == 1


def test_qt_strategy_survives_llm_failure(corpus) -> None:  # type: ignore[no-untyped-def]
    engine = _engine(corpus, KeywordReranker(), FakePlanner(error=ConnectionError("down")))
    result = engine.retrieve("GraphRAG limitations", strategy=RetrievalStrategy.HYBRID_QT_RERANK)
    assert result.queries == ["GraphRAG limitations"] and result.chunks
    assert result.transformation.error


def test_rerank_strategy_without_reranker_fails_clearly(corpus) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RerankerUnavailableError):
        _engine(corpus).retrieve("x query", strategy=RetrievalStrategy.HYBRID_RERANK)


# --- API ----------------------------------------------------------------------------------------


@pytest.fixture
def search_client(resources, corpus) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    from app.main import create_app

    client, embedder, encoder, _ = corpus
    resources.qdrant = client
    resources.dense_embedder = embedder
    resources.reranker = KeywordReranker()
    resources.settings.qdrant_collection = "chunks"
    with TestClient(create_app(resources=resources)) as test_client:
        yield test_client


def test_search_endpoint(search_client: TestClient) -> None:
    resp = search_client.post(
        "/api/v1/retrieval/search",
        json={"query": "GraphRAG limitations", "strategy": "hybrid_rerank", "top_k": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["strategy"] == "hybrid_rerank" and len(body["results"]) == 2
    hit = body["results"][0]
    assert hit["rank"] == 1 and hit["paper_id"] == PAPER_B
    assert {"rerank", "rrf"} <= set(hit["scores"]) and hit["page_start"] >= 1
    assert body["transformation"] is None


def test_search_endpoint_qt_without_llm_degrades(search_client: TestClient) -> None:
    resp = search_client.post(
        "/api/v1/retrieval/search", json={"query": "GraphRAG", "strategy": "hybrid_qt_rerank"}
    )
    assert resp.status_code == 200
    assert resp.json()["queries"] == ["GraphRAG"]
    assert resp.json()["transformation"]["used_llm"] is False


@pytest.mark.parametrize(
    "body",
    [
        {"query": "  "},
        {"query": "ok query", "top_k": 0},
        {"query": "ok query", "strategy": "magic"},
        {"query": "ok query", "year_from": 2024, "year_to": 2020},
    ],
)
def test_search_validation(search_client: TestClient, body: dict) -> None:
    assert search_client.post("/api/v1/retrieval/search", json=body).status_code == 422


def test_unknown_collection_is_404(search_client: TestClient) -> None:
    resp = search_client.post(
        "/api/v1/retrieval/search", json={"query": "abc", "collection": "nope"}
    )
    assert resp.status_code == 404
