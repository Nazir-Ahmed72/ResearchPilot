"""Durable LangGraph checkpoints on the real Neon database (writes then deletes one thread).
Opt-in: RUN_LIVE_TESTS=1 pytest tests/integration/test_live_checkpoint.py"""

from __future__ import annotations

import os
import uuid

import pytest

from app.core.config import Settings
from app.orchestration.checkpoint import create_checkpointer
from app.orchestration.graph import build_research_graph
from tests.fakes import ScriptedStep
from tests.integration.test_research_graph import Harness, good_synthesis

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1"),
]


def test_crash_then_resume_from_neon_in_a_new_process(corpus) -> None:  # type: ignore[no-untyped-def]
    settings = Settings()
    if settings.is_sqlite:
        pytest.skip("DATABASE_URL is not PostgreSQL")
    thread = f"test-{uuid.uuid4()}"

    saver1, close1 = create_checkpointer(settings)
    h = Harness(
        corpus,
        checkpointer=saver1,
        synthesizer=ScriptedStep(RuntimeError("worker died mid-synthesis")),
    )
    h.thread = thread
    try:
        with pytest.raises(RuntimeError):
            h.run()
    finally:
        close1()

    # "New process": a fresh pool/saver and a fresh graph, same thread id.
    saver2, close2 = create_checkpointer(settings)
    try:
        h.synthesizer.outputs = [good_synthesis]
        graph2 = build_research_graph(h.deps, checkpointer=saver2)
        config = {"configurable": {"thread_id": thread}}
        assert graph2.get_state(config).next == ("synthesize",)
        reranks = h.reranker.calls
        out = graph2.invoke(None, config)
        assert out["final"]["references"]
        assert h.reranker.calls == reranks  # nothing before synthesis was re-executed
    finally:
        saver2.delete_thread(thread)
        assert not graph2.get_state({"configurable": {"thread_id": thread}}).values
        close2()
