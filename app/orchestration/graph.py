"""Builds the LangGraph research workflow.

    START -> understand_query -> (simple) retrieve | (complex) plan_queries -> retrieve
    retrieve -> rerank -> (simple) synthesize | respond      (complex) evaluate_evidence
    evaluate_evidence -> plan_queries (insufficient, retries left) | synthesize | respond
    synthesize -> verify_citations -> synthesize (one repair) | respond -> END

Every node is wrapped by ``_instrument``, which (1) reports progress to an optional callback in
``config["configurable"]["on_node_start"]``, (2) enforces the run's overall deadline
(``config["configurable"]["deadline"]``, epoch seconds) before the node starts, and (3) records
per-node timings and a compact trace in the state.

Timeouts: LangGraph's per-node ``timeout`` only works for async nodes, and every component here
is synchronous (CPU models, sync DB/Qdrant clients). So individual calls carry their own timeouts
(LLM 60 s, Qdrant 30 s) and the wrapper enforces a whole-run deadline between nodes.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from app.orchestration import routing
from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.nodes.evidence import evaluate_evidence
from app.orchestration.nodes.planning import plan_queries
from app.orchestration.nodes.response import respond
from app.orchestration.nodes.retrieval import rerank, retrieve
from app.orchestration.nodes.synthesis import synthesize
from app.orchestration.nodes.understanding import understand_query
from app.orchestration.nodes.verification import verify_citations
from app.orchestration.state import ResearchState

NodeFn = Callable[[ResearchState, RunnableConfig, ResearchDependencies], dict[str, Any]]
NODE_ORDER = (
    "understand_query",
    "plan_queries",
    "retrieve",
    "rerank",
    "evaluate_evidence",
    "synthesize",
    "verify_citations",
    "respond",
)


class ResearchTimeoutError(TimeoutError):
    """The run exceeded its overall time budget. Not retried; the run can be resumed."""


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ResearchTimeoutError):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError, httpx.TransportError)):
        return True
    name = type(exc).__name__
    # qdrant-client and groq raise their own transport/rate-limit types.
    return name in {
        "ResponseHandlingException",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def is_retryable_llm_error(exc: BaseException) -> bool:
    if is_transient(exc) or isinstance(exc, (OutputParserException, ValidationError, TypeError)):
        return True
    # Groq rejects tool calls that violate the schema server-side (HTTP 400 tool_use_failed);
    # generation is stochastic, so a second attempt usually produces a valid call.
    return type(exc).__name__ == "BadRequestError" and "tool_use_failed" in str(exc)


def _instrument(name: str, fn: NodeFn, deps: ResearchDependencies) -> Callable[..., dict[str, Any]]:
    def node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        callback = configurable.get("on_node_start")
        if callback is not None:
            callback(name)
        # The deadline lives in the per-invocation config, so a resumed run gets a fresh budget
        # without rewriting checkpointed state.
        deadline = configurable.get("deadline")
        if deadline is not None and time.time() > deadline:
            raise ResearchTimeoutError(f"Research run exceeded its time budget before '{name}'")
        start = time.perf_counter()
        update = fn(state, config, deps)
        ms = round((time.perf_counter() - start) * 1000, 1)
        trace = {"node": name, "ms": ms, **update.pop("_trace", {})}
        return {**update, "node_timings": {name: ms}, "trace": [trace]}

    node.__name__ = name
    return node


def build_research_graph(
    deps: ResearchDependencies,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: Sequence[str] = (),
) -> CompiledStateGraph:
    unknown = set(interrupt_before) - set(NODE_ORDER)
    if unknown:
        raise ValueError(f"Unknown nodes in interrupt_before: {sorted(unknown)}")

    io_retry = RetryPolicy(max_attempts=3, initial_interval=1.0, retry_on=is_transient)
    llm_retry = RetryPolicy(max_attempts=2, initial_interval=2.0, retry_on=is_retryable_llm_error)

    graph = StateGraph(ResearchState)
    graph.add_node("understand_query", _instrument("understand_query", understand_query, deps))
    graph.add_node("plan_queries", _instrument("plan_queries", plan_queries, deps))
    graph.add_node("retrieve", _instrument("retrieve", retrieve, deps), retry_policy=io_retry)
    graph.add_node("rerank", _instrument("rerank", rerank, deps), retry_policy=io_retry)
    graph.add_node("evaluate_evidence", _instrument("evaluate_evidence", evaluate_evidence, deps))
    graph.add_node(
        "synthesize", _instrument("synthesize", synthesize, deps), retry_policy=llm_retry
    )
    graph.add_node("verify_citations", _instrument("verify_citations", verify_citations, deps))
    graph.add_node("respond", _instrument("respond", respond, deps))

    graph.add_edge(START, "understand_query")
    graph.add_conditional_edges("understand_query", routing.route_after_understanding)
    graph.add_edge("plan_queries", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", routing.route_after_rerank)
    graph.add_conditional_edges(
        "evaluate_evidence",
        functools.partial(
            routing.route_after_evaluation, max_retries=deps.config.max_retrieval_retries
        ),
        ["plan_queries", "synthesize", "respond"],
    )
    graph.add_edge("synthesize", "verify_citations")
    graph.add_conditional_edges("verify_citations", routing.route_after_verification)
    graph.add_edge("respond", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before),
        name="research_workflow",
    )
