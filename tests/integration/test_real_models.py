"""Real model checks (BGE embedder ~130 MB, BGE reranker ~1.1 GB). Opt-in: RUN_MODEL_TESTS=1"""

from __future__ import annotations

import math
import os

import pytest

from app.core.config import PROJECT_ROOT
from app.embeddings.dense import SentenceTransformerEmbedder

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run"
    ),
]


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    # Shared project cache, so the model is downloaded once (to E:), not per test run.
    return SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", cache_dir=PROJECT_ROOT / ".cache/hf"
    )


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_model_properties(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.dimension == 384
    assert embedder.max_tokens == 512
    assert embedder.count_tokens("retrieval augmented generation") > 0


def test_vectors_are_normalised(embedder: SentenceTransformerEmbedder) -> None:
    vectors = embedder.embed_documents(["first passage", "second passage"])
    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == 384
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-3)


def test_relevant_passage_ranks_above_irrelevant(embedder: SentenceTransformerEmbedder) -> None:
    query = embedder.embed_query("How does retrieval-augmented generation reduce hallucination?")
    relevant, irrelevant = embedder.embed_documents(
        [
            "RAG conditions the generator on retrieved documents, which grounds outputs in "
            "evidence and reduces hallucinated facts.",
            "The recipe calls for two cups of flour, sugar, and a pinch of salt.",
        ]
    )
    assert _cos(query, relevant) > _cos(query, irrelevant) + 0.2


@pytest.fixture(scope="module")
def reranker():  # type: ignore[no-untyped-def]
    from app.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker("BAAI/bge-reranker-base", cache_dir=PROJECT_ROOT / ".cache/hf")


def test_reranker_orders_relevance_correctly(reranker) -> None:  # type: ignore[no-untyped-def]
    # Reference pair from the BGE model card: clearly relevant vs. clearly irrelevant.
    panda = reranker.score(
        "what is panda?",
        ["hi", "The giant panda (Ailuropoda melanoleuca) is a bear species endemic to China."],
    )
    assert panda[1] > 0.9 and panda[0] < 0.01

    # Scores are monotonic but NOT calibrated: a relevant passage that never names the query's
    # subject ("RAG") still scores low in absolute terms, yet ranks above an irrelevant one.
    scores = reranker.score(
        "Which retriever does RAG use?",
        [
            "The retrieval component is based on DPR, a bi-encoder dense passage retriever.",
            "We thank the reviewers for their helpful comments.",
        ],
    )
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1] * 10
