"""Pure retrieval logic: RRF fusion, reranking order, query transformation validation."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query_transformer import QueryPlan, QueryTransformer, build_queries
from app.retrieval.reranker import _sigmoid, rerank
from app.retrieval.types import RetrievedChunk
from tests.fakes import FakePlanner, KeywordReranker


def chunk(
    cid: str, stage: str = "dense", score: float = 1.0, text: str = "", rank: int = 1
) -> RetrievedChunk:
    return RetrievedChunk(
        cid, {"text": text or cid, "paper_id": "p"}, score, {stage: score}, {stage: rank}
    )


# --- RRF ----------------------------------------------------------------------------------------


def test_rrf_rewards_agreement_between_lists() -> None:
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("c", "sparse"), chunk("d", "sparse"), chunk("a", "sparse")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [c.chunk_id for c in fused]
    assert ids[:2] == ["a", "c"]  # in both lists
    assert set(ids) == {"a", "b", "c", "d"}  # deduplicated union
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_keeps_scores_from_every_stage() -> None:
    fused = reciprocal_rank_fusion([[chunk("a", "dense", 0.8)], [chunk("a", "sparse", 12.5)]], k=60)
    (a,) = fused
    assert a.scores["dense"] == 0.8 and a.scores["sparse"] == 12.5
    assert a.ranks == {"dense": 1, "sparse": 1, "rrf": 1}


def test_rrf_list_cannot_vote_twice_for_a_chunk() -> None:
    fused = reciprocal_rank_fusion([[chunk("a"), chunk("a")]], k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_weights_shift_the_ranking() -> None:
    dense, sparse = [chunk("a"), chunk("b")], [chunk("b", "sparse"), chunk("a", "sparse")]
    assert reciprocal_rank_fusion([dense, sparse], weights=[2, 1])[0].chunk_id == "a"
    assert reciprocal_rank_fusion([dense, sparse], weights=[1, 2])[0].chunk_id == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([dense], weights=[1, 2])


def test_rrf_ties_are_deterministic_and_empty_input_is_fine() -> None:
    fused = reciprocal_rank_fusion([[chunk("z")], [chunk("y")]])
    assert [c.chunk_id for c in fused] == ["y", "z"]
    assert reciprocal_rank_fusion([]) == [] and reciprocal_rank_fusion([[], []]) == []


# --- reranking ----------------------------------------------------------------------------------


def test_rerank_reorders_and_truncates_preserving_metadata() -> None:
    candidates = [
        chunk("x", text="weather report for tuesday"),
        chunk("y", text="dense retrieval with dual encoders improves recall"),
        chunk("z", text="retrieval benchmark"),
    ]
    out = rerank(KeywordReranker(), "how does dense retrieval improve recall", candidates, 2)
    assert [c.chunk_id for c in out] == ["y", "z"]
    assert out[0].scores["rerank"] > out[1].scores["rerank"]
    assert out[0].scores["dense"] == 1.0  # earlier stage scores preserved
    assert out[0].ranks["rerank"] == 1 and out[0].payload["paper_id"] == "p"


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    assert _sigmoid(0) == 0.5
    assert _sigmoid(1000) == pytest.approx(1.0) and _sigmoid(-1000) == pytest.approx(0.0)


# --- query transformation ------------------------------------------------------------------------


def plan(**kw: object) -> QueryPlan:
    base = {"complexity": "simple", "rewritten_query": "retrieval augmented generation definition"}
    return QueryPlan(**{**base, **kw})


def test_simple_question_gets_only_a_rewrite() -> None:
    p = plan(alternative_queries=["RAG overview"], sub_questions=["what is retrieval"])
    queries = build_queries("What is RAG?", p, max_queries=4)
    assert queries == ["What is RAG?", "retrieval augmented generation definition"]


def test_complex_question_is_decomposed_deduplicated_and_capped() -> None:
    p = plan(
        complexity="complex",
        rewritten_query="Compare RAG and GraphRAG",
        sub_questions=[
            "How does RAG retrieve?",
            "how does rag retrieve?",
            "How does GraphRAG work?",
        ],
        alternative_queries=["  RAG   vs  GraphRAG  ", "x", "graph-based retrieval augmentation"],
    )
    queries = build_queries("Compare RAG and GraphRAG", p, max_queries=4)
    assert queries == [
        "Compare RAG and GraphRAG",
        "How does RAG retrieve?",
        "How does GraphRAG work?",
        "RAG vs GraphRAG",
    ]


def test_invalid_generated_queries_are_dropped() -> None:
    p = plan(complexity="complex", rewritten_query="!!!", sub_questions=["", "ok?", "y" * 400])
    assert build_queries("original question", p, max_queries=4) == ["original question"]


def test_transformer_uses_planner_output() -> None:
    planner = FakePlanner(plan(complexity="complex", sub_questions=["What are RAG limitations?"]))
    result = QueryTransformer(planner, max_queries=4).transform("  RAG   limits?  ")
    assert planner.inputs == [{"question": "RAG limits?"}]
    assert result.used_llm and result.complexity == "complex"
    assert result.queries[0] == "RAG limits?"
    assert "What are RAG limitations?" in result.queries


@pytest.mark.parametrize(
    "planner",
    [
        FakePlanner(error=TimeoutError("groq timed out; api_key=gsk_abcdefghijklmnop")),
        FakePlanner(plan="not a plan"),
        None,
    ],
)
def test_transformer_falls_back_to_original_query(planner: FakePlanner | None) -> None:
    result = QueryTransformer(planner).transform("What is RAG?")
    assert result.queries == ["What is RAG?"]
    assert not result.used_llm and result.error
    assert "gsk_abcdefghijklmnop" not in result.error
