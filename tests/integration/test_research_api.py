"""Research endpoints end to end: submit -> background run (inline) -> poll -> answer.

Uses SQLite, the in-memory Qdrant corpus, fake LLM steps and an in-memory checkpointer.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.core.resources import Resources
from app.database.models import Paper, PaperCollection, PaperStatus
from app.main import create_app
from app.retrieval.query_transformer import QueryPlan
from app.workers.dispatch import InlineDispatcher
from tests.corpus import PAPER_A, PAPER_B
from tests.fakes import FakePlanner, KeywordReranker, ScriptedStep
from tests.integration.test_research_graph import (
    COMPARE,
    SIMPLE,
    SUFFICIENT,
    all_supported,
    good_synthesis,
)

QUERY = "/api/v1/research/query"


def _steps(synthesizer=None) -> dict:  # type: ignore[no-untyped-def]
    return {
        "analyzer": ScriptedStep(SIMPLE),
        "evaluator": ScriptedStep(SUFFICIENT),
        "synthesizer": synthesizer or ScriptedStep(good_synthesis),
        "verifier": ScriptedStep(all_supported),
    }


@pytest.fixture
def research_env(resources: Resources, corpus) -> Iterator[tuple[TestClient, Resources]]:  # type: ignore[no-untyped-def]
    client, embedder, _, _ = corpus
    resources.qdrant = client
    resources.dense_embedder = embedder
    resources.reranker = KeywordReranker()
    resources.settings.qdrant_collection = "chunks"
    resources._query_planner = FakePlanner(QueryPlan(complexity="simple", rewritten_query="x y z"))
    resources._llm_steps = _steps()
    resources._checkpointer = InMemorySaver()
    resources._dispatcher = InlineDispatcher(
        resources.ingestion_pipeline, lambda rid: resources.research_runner.run(rid)
    )
    with resources.session_factory() as s:
        coll = PaperCollection(name="rag")
        s.add(coll)
        for pid, title in ((PAPER_A, "Paper A"), (PAPER_B, "Paper B")):
            s.add(
                Paper(
                    id=uuid.UUID(pid),
                    collection=coll,
                    content_hash=pid.replace("-", "")[:64],
                    original_filename=f"{pid}.pdf",
                    storage_path=f"{pid}.pdf",
                    file_size_bytes=1,
                    title=title,
                    status=PaperStatus.READY,
                )
            )
        s.commit()
    with TestClient(create_app(resources=resources)) as test_client:
        yield test_client, resources


def _wait(resources: Resources) -> None:
    for future in resources.dispatcher.futures.values():  # type: ignore[attr-defined]
        future.result(timeout=60)


def test_submit_poll_and_get_answer(research_env) -> None:  # type: ignore[no-untyped-def]
    client, resources = research_env
    resp = client.post(QUERY, json={"question": "What does GraphRAG build?"})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/api/v1/research/{body['request_id']}"
    _wait(resources)

    status = client.get(body["status_url"]).json()
    assert status["status"] == "succeeded" and status["current_node"] is None
    assert status["analysis_type"] == "factual"
    result = status["result"]
    assert result["references"] and result["references"][0]["number"] == 1
    assert "## References" in result["answer_markdown"]
    assert result["evidence_sufficient"] is True
    assert result["verification"]["citation_validity"] == 1.0
    m = result["metrics"]
    assert m["latency_ms"] > 0 and "synthesize" in m["node_timings_ms"]
    assert m["retry_count"] == 0 and m["cost_usd"] is None
    assert m["models"]["main"]


def test_analysis_type_override_is_respected(research_env) -> None:  # type: ignore[no-untyped-def]
    client, resources = research_env
    rid = client.post(
        QUERY,
        json={"question": "Tell me about these retrieval methods", "analysis_type": "comparison"},
    ).json()["request_id"]
    _wait(resources)
    status = client.get(f"/api/v1/research/{rid}").json()
    assert status["analysis_type"] == "comparison" and status["status"] == "succeeded"


def test_failed_run_can_be_resumed_from_checkpoint(research_env) -> None:  # type: ignore[no-untyped-def]
    client, resources = research_env
    synth = ScriptedStep(RuntimeError("model crashed"), good_synthesis)
    resources._llm_steps = _steps(synth)
    resources.__dict__.pop("research_graph", None)
    resources.__dict__.pop("research_runner", None)

    rid = client.post(QUERY, json={"question": "What does GraphRAG build?"}).json()["request_id"]
    _wait(resources)
    failed = client.get(f"/api/v1/research/{rid}").json()
    assert failed["status"] == "failed" and "model crashed" in failed["error_message"]
    assert failed["current_node"] == "synthesize"

    calls_before = resources.reranker.calls
    resp = client.post(f"/api/v1/research/{rid}/resume")
    assert resp.status_code == 202
    _wait(resources)
    done = client.get(f"/api/v1/research/{rid}").json()
    assert done["status"] == "succeeded" and done["result"]["references"]
    assert resources.reranker.calls == calls_before  # resumed: retrieval not repeated
    assert client.post(f"/api/v1/research/{rid}/resume").status_code == 409


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"question": "Hi?"}, 422),
        ({"question": "valid question here", "strategy": "nope"}, 422),
        ({"question": "valid question here", "collection": "missing"}, 404),
        (
            {
                "question": "valid question here",
                "paper_ids": ["00000000-0000-0000-0000-000000000000"],
            },
            404,
        ),
    ],
)
def test_submit_validation(research_env, body: dict, status: int) -> None:  # type: ignore[no-untyped-def]
    client, _ = research_env
    assert client.post(QUERY, json=body).status_code == status


def test_no_ready_papers_in_scope_is_rejected(research_env) -> None:  # type: ignore[no-untyped-def]
    client, resources = research_env
    with resources.session_factory() as s:
        s.get(Paper, uuid.UUID(PAPER_A)).status = PaperStatus.PROCESSING
        s.commit()
    resp = client.post(QUERY, json={"question": "valid question here", "paper_ids": [PAPER_A]})
    assert resp.status_code == 409


def test_unknown_request_is_404(research_env) -> None:  # type: ignore[no-untyped-def]
    client, _ = research_env
    assert client.get(f"/api/v1/research/{uuid.uuid4()}").status_code == 404
    assert client.post(f"/api/v1/research/{uuid.uuid4()}/resume").status_code == 404


def test_comparison_uses_both_papers(research_env) -> None:  # type: ignore[no-untyped-def]
    client, resources = research_env
    resources._llm_steps = {**_steps(), "analyzer": ScriptedStep(COMPARE)}
    resources.__dict__.pop("research_graph", None)
    resources.__dict__.pop("research_runner", None)
    rid = client.post(QUERY, json={"question": "Compare RAG and GraphRAG"}).json()["request_id"]
    _wait(resources)
    with resources.session_factory() as s:
        from app.database.models import ResearchRequest

        result = s.get(ResearchRequest, uuid.UUID(rid)).result
        trace_nodes = [t["node"] for t in result.metrics["trace"]]
    assert "evaluate_evidence" in trace_nodes and "plan_queries" in trace_nodes


def test_collection_scope_reaches_retrieval_filters(research_env) -> None:  # type: ignore[no-untyped-def]
    """The fake corpus tags chunks with other collection ids, so a scoped run must find nothing
    and answer transparently instead of using out-of-scope evidence."""
    client, resources = research_env
    rid = client.post(
        QUERY, json={"question": "What does GraphRAG build?", "collection": "rag"}
    ).json()["request_id"]
    _wait(resources)
    result = client.get(f"/api/v1/research/{rid}").json()["result"]
    assert result["references"] == [] and result["evidence_sufficient"] is False
    assert "No passages relevant" in result["answer_markdown"]
