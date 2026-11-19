"""End-to-end evaluation of the research workflow.

Per question: did the run complete, how long did it take, how many tokens did it use, how good
are its citations, and did it abstain correctly?

* Citation validity / coverage / support rate come from the workflow's own verification
  (support rate is an LLM judgement).
* **Citation label accuracy** is independent of any LLM: the share of cited references whose
  chunk satisfies one of the question's evidence labels.
* **Evidence recall**: the share of evidence labels covered by the evidence the workflow used.
* **Abstention**: unanswerable questions should end with ``evidence_sufficient = false``;
  answerable ones should not (false abstention).

LLM judging (verification, Ragas) is useful but not ground truth: the judge can be wrong, is
sensitive to prompts, and may favour fluent answers. The label-based metrics are the anchor.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langgraph.graph.state import CompiledStateGraph

from app.evaluation.dataset import EvalDataset, EvalQuestion, matches_label
from app.evaluation.metrics import mean, percentile

logger = logging.getLogger(__name__)


def answer_text(final: Mapping[str, Any]) -> str:
    """The answer as plain text for LLM judges: summary + kept claims (no reference list)."""
    parts = [final.get("summary") or ""]
    for section in final.get("sections") or []:
        parts += [c.get("text", "") for c in section.get("claims", [])]
    return "\n".join(p for p in parts if p).strip()


def score_question(
    q: EvalQuestion, values: Mapping[str, Any], dataset: EvalDataset
) -> dict[str, Any]:
    final = values.get("final") or {}
    evidence = values.get("evidence") or []
    by_chunk = {e["chunk_id"]: e for e in evidence}
    refs = final.get("references") or []

    def relevant(chunk: Mapping[str, Any] | None) -> bool:
        return bool(chunk) and any(matches_label(chunk, lab, dataset.papers) for lab in q.evidence)

    cited_relevant = [relevant(by_chunk.get(r["chunk_id"])) for r in refs]
    covered = {
        i
        for i, lab in enumerate(q.evidence)
        if any(matches_label(e, lab, dataset.papers) for e in evidence)
    }
    verification = final.get("verification") or {}
    sufficient = bool(final.get("evidence_sufficient"))
    return {
        "evidence_sufficient": sufficient,
        "abstained_correctly": (not sufficient) if not q.answerable else None,
        "false_abstention": (not sufficient) if q.answerable else None,
        "citation_label_accuracy": (
            round(sum(cited_relevant) / len(cited_relevant), 4)
            if cited_relevant and q.answerable
            else None
        ),
        "evidence_recall": round(len(covered) / len(q.evidence), 4) if q.evidence else None,
        "references": len(refs),
        "removed_statements": len(final.get("removed_statements") or []),
        "citation_validity": verification.get("citation_validity"),
        "citation_coverage": verification.get("citation_coverage"),
        "support_rate": verification.get("support_rate"),
        "retry_count": values.get("retry_count", 0),
        "repair_count": values.get("repair_count", 0),
        "analysis_type": (values.get("analysis") or {}).get("analysis_type"),
    }


def evaluate_generation(
    graph: CompiledStateGraph,
    dataset: EvalDataset,
    *,
    strategy: str,
    filters: Mapping[str, Any] | None = None,
    tracer: BaseCallbackHandler | None = None,
    pace_seconds: float = 0.0,
    timeout_seconds: float = 300.0,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Runs every question through the workflow (no checkpointer). ``pace_seconds`` waits
    between questions to stay under provider rate limits (Groq free tier: 8k tokens/min)."""
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(dataset.questions):
        if i and pace_seconds:
            time.sleep(pace_seconds)
        if on_progress:
            on_progress(q.id)
        usage = UsageMetadataCallbackHandler()
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": f"eval-{uuid.uuid4()}",
                "deadline": time.time() + timeout_seconds,
            },
            "callbacks": [usage, *([tracer] if tracer else [])],
            "run_name": f"eval:{q.id}",
            "tags": ["evaluation", strategy, q.category],
            "metadata": {
                "question_id": q.id,
                "category": q.category,
                "strategy": strategy,
                "dataset": dataset.name,
            },
            "recursion_limit": 50,
        }
        state = {
            "request_id": f"eval-{q.id}",
            "question": q.question,
            "strategy": strategy,
            "filters": dict(filters or {}),
        }
        start = time.perf_counter()
        row: dict[str, Any] = {
            "question_id": q.id,
            "category": q.category,
            "question": q.question,
            "answerable": q.answerable,
        }
        try:
            values = graph.invoke(state, config)
        except Exception as exc:  # noqa: BLE001 - record and continue with the next question
            logger.warning("Generation run failed", extra={"question": q.id}, exc_info=True)
            rows.append(
                {
                    **row,
                    "completed": False,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                }
            )
            continue
        final = values.get("final") or {}
        rows.append(
            {
                **row,
                "completed": True,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "node_timings_ms": values.get("node_timings") or {},
                "token_usage": {m: dict(u) for m, u in usage.usage_metadata.items()},
                "answer": answer_text(final),
                "answer_markdown": final.get("answer_markdown", ""),
                "contexts": [e["text"] for e in values.get("evidence") or []],
                "reference_answer": q.reference_answer,
                "errors": values.get("errors") or [],
                **score_question(q, values, dataset),
            }
        )
    return rows


def total_tokens(row: Mapping[str, Any]) -> int:
    return sum(int(u.get("total_tokens", 0)) for u in (row.get("token_usage") or {}).values())


def summarize_generation(
    rows: Sequence[Mapping[str, Any]], pricing: Mapping[str, Mapping[str, float]] | None = None
) -> dict[str, Any]:
    done = [r for r in rows if r.get("completed")]
    answerable = [r for r in done if r["answerable"]]
    unanswerable = [r for r in done if not r["answerable"]]
    latencies = [r["latency_ms"] for r in done if r.get("latency_ms") is not None]
    node_ms: dict[str, list[float]] = {}
    for r in done:
        for node, ms in (r.get("node_timings_ms") or {}).items():
            node_ms.setdefault(node, []).append(ms)

    cost = None
    if pricing and done:
        try:
            cost = round(
                sum(
                    u.get("input_tokens", 0) / 1e6 * pricing[m]["input"]
                    + u.get("output_tokens", 0) / 1e6 * pricing[m]["output"]
                    for r in done
                    for m, u in (r.get("token_usage") or {}).items()
                )
                / len(done),
                6,
            )
        except KeyError:
            cost = None  # a model without a configured price: do not report a partial cost

    ragas_keys = sorted({k for r in done for k in (r.get("ragas") or {})})
    return {
        "questions": len(rows),
        "completed": len(done),
        "completion_rate": round(len(done) / len(rows), 4) if rows else None,
        "abstention_accuracy": mean(
            1.0 if r.get("abstained_correctly") else 0.0 for r in unanswerable
        ),
        "false_abstention_rate": mean(
            1.0 if r.get("false_abstention") else 0.0 for r in answerable
        ),
        "citation_label_accuracy": mean(r.get("citation_label_accuracy") for r in answerable),
        "evidence_recall": mean(r.get("evidence_recall") for r in answerable),
        "citation_validity": mean(r.get("citation_validity") for r in answerable),
        "citation_coverage": mean(r.get("citation_coverage") for r in answerable),
        "support_rate": mean(r.get("support_rate") for r in answerable),
        "retry_rate": mean(1.0 if r.get("retry_count") else 0.0 for r in done),
        "repair_rate": mean(1.0 if r.get("repair_count") else 0.0 for r in done),
        "latency_ms_mean": mean(latencies),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "node_latency_ms_mean": {n: mean(v) for n, v in sorted(node_ms.items())},
        "tokens_per_query_mean": mean(total_tokens(r) for r in done),
        "cost_usd_per_query": cost,
        "ragas": {k: mean((r.get("ragas") or {}).get(k) for r in done) for k in ragas_keys},
    }
