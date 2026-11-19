"""The LangGraph research workflow end to end, with scripted LLM steps and a real in-memory
Qdrant corpus (the 6-chunk, 2-paper corpus from test_retrieval)."""

from __future__ import annotations

import time
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.orchestration.dependencies import ResearchConfig, ResearchDependencies
from app.orchestration.graph import ResearchTimeoutError, build_research_graph
from app.orchestration.schemas import (
    Claim,
    ClaimVerdict,
    EvidenceAssessment,
    QueryAnalysis,
    Synthesis,
    SynthesisSection,
    VerificationResult,
)
from app.retrieval.engine import RetrievalConfig, RetrievalEngine
from app.retrieval.query_transformer import QueryPlan, QueryTransformer
from app.retrieval.retrievers import DenseRetriever, HybridRetriever, SparseRetriever
from tests.corpus import PAPER_A, PAPER_B
from tests.fakes import FakePlanner, KeywordReranker, ScriptedStep

SIMPLE = QueryAnalysis(
    analysis_type="factual",
    complexity="simple",
    requires_cross_paper=False,
    key_concepts=["GraphRAG"],
)
COMPARE = QueryAnalysis(
    analysis_type="comparison",
    complexity="complex",
    requires_cross_paper=True,
    key_concepts=["RAG", "GraphRAG"],
)
SUFFICIENT = EvidenceAssessment(sufficient=True)
INSUFFICIENT = EvidenceAssessment(
    sufficient=False,
    missing_aspects=["cost of indexing"],
    follow_up_queries=["GraphRAG index construction cost"],
)


def good_synthesis(_: dict | None = None) -> Synthesis:
    return Synthesis(
        summary="GraphRAG summarises graph communities [E1].",
        sections=[
            SynthesisSection(
                heading="Explanation",
                claims=[
                    Claim(
                        text="GraphRAG builds a knowledge graph of entities.", evidence_ids=["E1"]
                    ),
                    Claim(text="Graph-based indexing may be costly at scale.", basis="inferred"),
                ],
            )
        ],
        uncertainty="Only two papers were available.",
    )


def all_supported(inp: dict) -> VerificationResult:
    ids = [
        line.split(":")[0].split()[1]
        for line in inp["claims"].splitlines()
        if line.startswith("Claim C")
    ]
    return VerificationResult(verdicts=[ClaimVerdict(claim_id=i, verdict="supported") for i in ids])


class Harness:
    def __init__(
        self,
        corpus,
        *,
        analyzer=None,
        evaluator=None,
        synthesizer=None,
        verifier=None,
        planner=None,
        reranker=None,
        config=None,
        checkpointer=None,
        interrupt_before=(),
    ):
        _, embedder, encoder, searcher = corpus
        self.reranker = reranker or KeywordReranker()
        engine = RetrievalEngine(
            dense=DenseRetriever(searcher, embedder),
            sparse=SparseRetriever(searcher, encoder),
            hybrid=HybridRetriever(searcher, embedder, encoder),
            reranker=self.reranker,
            transformer=None,
            config=RetrievalConfig(top_k=3, candidate_k=10),
        )
        self.analyzer = analyzer if analyzer is not None else ScriptedStep(SIMPLE)
        self.evaluator = evaluator if evaluator is not None else ScriptedStep(SUFFICIENT)
        self.synthesizer = synthesizer if synthesizer is not None else ScriptedStep(good_synthesis)
        self.verifier = verifier if verifier is not None else ScriptedStep(all_supported)
        planner = planner or FakePlanner(
            QueryPlan(
                complexity="complex",
                rewritten_query="RAG versus GraphRAG",
                sub_questions=["How does GraphRAG index documents?"],
            )
        )
        self.deps = ResearchDependencies(
            engine=engine,
            transformer=QueryTransformer(planner, max_queries=4),
            analyzer=self.analyzer,
            evaluator=self.evaluator,
            synthesizer=self.synthesizer,
            verifier=self.verifier,
            config=config
            or ResearchConfig(
                evidence_k_simple=3,
                evidence_k_complex=4,
                candidate_k_simple=6,
                candidate_k_complex=6,
                max_chunks_per_paper=2,
            ),
        )
        self.graph = build_research_graph(
            self.deps, checkpointer=checkpointer, interrupt_before=interrupt_before
        )
        self.thread = str(uuid.uuid4())

    def config(self, **configurable):
        return {"configurable": {"thread_id": self.thread, **configurable}}

    def run(self, question="What does GraphRAG build?", strategy="hybrid_rerank", **extra):
        state = {
            "request_id": self.thread,
            "question": question,
            "strategy": strategy,
            "filters": {},
            **extra,
        }
        return self.graph.invoke(state, self.config())


def path(state) -> list[str]:
    return [t["node"] for t in state["trace"]]


# --- happy paths ------------------------------------------------------------------------------


def test_simple_question_follows_short_path(corpus) -> None:
    h = Harness(corpus)
    out = h.run()
    assert path(out) == [
        "understand_query",
        "retrieve",
        "rerank",
        "synthesize",
        "verify_citations",
        "respond",
    ]
    final = out["final"]
    assert final["evidence_sufficient"] is True
    assert final["references"][0]["number"] == 1 and final["references"][0]["page_start"]
    assert "[1]" in final["summary"] and "[E1]" not in final["answer_markdown"]
    assert "*Inferred:*" in final["answer_markdown"]  # inferred claim labelled
    assert final["verification"]["citation_validity"] == 1.0
    assert h.evaluator.inputs == []  # simple path skips evidence evaluation
    assert set(out["node_timings"]) == set(path(out))


def test_complex_question_plans_evaluates_and_retries_once(corpus) -> None:
    h = Harness(
        corpus, analyzer=ScriptedStep(COMPARE), evaluator=ScriptedStep(INSUFFICIENT, SUFFICIENT)
    )
    out = h.run("Compare RAG and GraphRAG")
    assert path(out) == [
        "understand_query",
        "plan_queries",
        "retrieve",
        "rerank",
        "evaluate_evidence",
        "plan_queries",
        "retrieve",
        "rerank",
        "evaluate_evidence",
        "synthesize",
        "verify_citations",
        "respond",
    ]
    assert out["retry_count"] == 1
    assert "GraphRAG index construction cost" in out["search_queries"]
    papers = {e["paper_id"] for e in out["evidence"]}
    assert papers == {PAPER_A, PAPER_B}  # cross-paper question draws on both papers
    counts = [sum(e["paper_id"] == p for e in out["evidence"]) for p in papers]
    assert max(counts) <= 2  # per-paper cap


def test_retries_are_bounded_and_insufficiency_is_transparent(corpus) -> None:
    h = Harness(corpus, analyzer=ScriptedStep(COMPARE), evaluator=ScriptedStep(INSUFFICIENT))
    out = h.run("Compare RAG and GraphRAG")
    assert path(out).count("plan_queries") == 2  # initial + exactly one retry
    assert out["retry_count"] == 1
    assert "INSUFFICIENT" in h.synthesizer.inputs[-1]["sufficiency_note"]
    assert out["final"]["evidence_sufficient"] is False
    assert "Evidence was insufficient" in out["final"]["answer_markdown"]


def test_no_evidence_answers_transparently_without_synthesis(corpus) -> None:
    h = Harness(corpus)
    out = h.run(filters={"paper_ids": ("99999999-9999-9999-9999-999999999999",)})
    assert path(out) == ["understand_query", "retrieve", "rerank", "respond"]
    assert h.synthesizer.inputs == []
    assert out["final"]["evidence_sufficient"] is False
    assert "No passages relevant" in out["final"]["answer_markdown"]


@pytest.mark.parametrize("strategy", ["dense", "sparse", "hybrid"])
def test_non_rerank_strategies_skip_the_cross_encoder(corpus, strategy: str) -> None:
    h = Harness(corpus)
    out = h.run(strategy=strategy)
    assert h.reranker.calls == 0 and out["final"]["references"]


def test_qt_strategy_plans_even_simple_questions(corpus) -> None:
    out = Harness(corpus).run(strategy="hybrid_qt_rerank")
    assert "plan_queries" in path(out) and len(out["search_queries"]) >= 2


# --- citation verification ----------------------------------------------------------------------


def bad_synthesis(_: dict | None = None) -> Synthesis:
    return Synthesis(
        summary="RAG is great [E1] [E99].",
        sections=[
            SynthesisSection(
                heading="Findings",
                claims=[
                    Claim(text="GraphRAG builds a knowledge graph.", evidence_ids=["E1"]),
                    Claim(
                        text="GraphRAG won a Turing award.", evidence_ids=[]
                    ),  # uncited, reported
                    Claim(text="It is 10x faster.", evidence_ids=["E42"]),  # fabricated citation
                ],
            )
        ],
    )


def test_bad_citations_trigger_one_repair(corpus) -> None:
    h = Harness(corpus, synthesizer=ScriptedStep(bad_synthesis, good_synthesis))
    out = h.run()
    assert path(out).count("synthesize") == 2 and out["repair_count"] == 1
    repair_note = h.synthesizer.inputs[1]["repair_note"]
    assert "Turing award" in repair_note and "E42" in repair_note
    assert out["final"]["removed_statements"] == []


def test_repairs_are_bounded_and_unsupported_claims_removed(corpus) -> None:
    h = Harness(corpus, synthesizer=ScriptedStep(bad_synthesis))
    out = h.run()
    assert path(out).count("synthesize") == 2  # one repair, then give up
    final = out["final"]
    removed = [r["text"] for r in final["removed_statements"]]
    assert "GraphRAG won a Turing award." in removed and "It is 10x faster." in removed
    assert "Turing" not in final["answer_markdown"].split("## Statements removed")[0]
    assert "[E99]" not in final["summary"] and final["verification"]["citation_validity"] < 1


def test_llm_judged_unsupported_claim_is_removed(corpus) -> None:
    def reject_c1(inp):
        return VerificationResult(
            verdicts=[
                ClaimVerdict(claim_id="C0", verdict="supported"),
                ClaimVerdict(
                    claim_id="C1", verdict="unsupported", reason="evidence says otherwise"
                ),
            ]
        )

    out = Harness(corpus, verifier=ScriptedStep(reject_c1)).run()
    assert out["final"]["removed_statements"][0]["reason"] == "evidence says otherwise"


# --- failure handling ---------------------------------------------------------------------------


def test_llm_failures_in_auxiliary_steps_degrade_gracefully(corpus) -> None:
    h = Harness(
        corpus,
        analyzer=ScriptedStep(ConnectionError("groq down; api_key=sk-secret12345")),
        verifier=ScriptedStep(RuntimeError("verifier broke")),
    )
    out = h.run("Compare RAG and GraphRAG approaches")
    assert out["analysis"]["source"] == "heuristic"
    assert out["analysis"]["analysis_type"] == "comparison"
    assert any("understand_query" in e for e in out["errors"])
    assert any("verify_citations" in e for e in out["errors"])
    assert not any("sk-secret12345" in e for e in out["errors"])
    assert "*(not verified)*" in out["final"]["answer_markdown"]


def test_deadline_is_enforced(corpus) -> None:
    h = Harness(corpus)
    state = {"request_id": "r", "question": "q?", "strategy": "hybrid", "filters": {}}
    with pytest.raises(ResearchTimeoutError, match="understand_query"):
        h.graph.invoke(state, h.config(deadline=time.time() - 1))


def test_transient_retrieval_errors_are_retried(corpus, monkeypatch) -> None:
    h = Harness(corpus)
    real = h.deps.engine.hybrid.search
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("qdrant blip")
        return real(*a, **kw)

    monkeypatch.setattr(h.deps.engine.hybrid, "search", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = h.run()
    assert calls["n"] == 2 and out["final"]["references"]


# --- checkpointing, resume, human-in-the-loop -----------------------------------------------------


def test_checkpoint_resume_continues_without_redoing_work(corpus) -> None:
    saver = InMemorySaver()
    h = Harness(
        corpus,
        checkpointer=saver,
        synthesizer=ScriptedStep(RuntimeError("synthesis crashed"), good_synthesis),
    )
    with pytest.raises(RuntimeError, match="synthesis crashed"):
        h.run()
    snapshot = h.graph.get_state(h.config())
    assert snapshot.next == ("synthesize",)
    assert snapshot.values["evidence"]  # retrieval results are in the checkpoint
    reranks_before = h.reranker.calls

    out = h.graph.invoke(None, h.config())  # resume from the last checkpoint
    assert out["final"]["references"]
    assert h.reranker.calls == reranks_before  # retrieval/rerank were not repeated
    assert path(out).count("retrieve") == 1


def test_interrupt_before_synthesis_allows_human_review(corpus) -> None:
    h = Harness(corpus, checkpointer=InMemorySaver(), interrupt_before=["synthesize"])
    partial = h.run()
    assert "final" not in partial
    assert h.graph.get_state(h.config()).next == ("synthesize",)
    out = h.graph.invoke(None, h.config())
    assert out["final"]["references"]


def test_progress_callback_sees_every_node(corpus) -> None:
    h = Harness(corpus)
    seen: list[str] = []
    h.graph.invoke(
        {
            "request_id": "r",
            "question": "What does GraphRAG build?",
            "strategy": "hybrid_rerank",
            "filters": {},
        },
        h.config(on_node_start=seen.append),
    )
    assert seen == [
        "understand_query",
        "retrieve",
        "rerank",
        "synthesize",
        "verify_citations",
        "respond",
    ]


def test_unknown_interrupt_node_rejected(corpus) -> None:
    with pytest.raises(ValueError):
        Harness(corpus, interrupt_before=["nope"])
