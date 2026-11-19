"""Research requests: creation and status (API side) and execution (worker side).

``ResearchService`` validates and records a request, then hands it to a dispatcher, so the API
never blocks on the multi-minute workflow. ``ResearchRunner`` executes the LangGraph workflow in
the worker, keeps ``current_node`` updated for progress polling, and persists the result.

Resumability: the LangGraph thread id is the request id. When a run is re-executed (resume
endpoint, or Celery redelivering a task after a worker crash) and a checkpoint with pending
nodes exists, the runner continues from that checkpoint instead of starting over.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.core.config import Settings
from app.core.exceptions import ConflictError, DependencyUnavailableError, NotFoundError
from app.core.logging import redact
from app.core.types import AnalysisType, RetrievalStrategy
from app.database.models import (
    JobStatus,
    Paper,
    PaperStatus,
    ResearchRequest,
    ResearchResult,
)
from app.database.repositories.papers import CollectionRepository
from app.database.session import session_scope
from app.workers.dispatch import TaskDispatcher

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 50  # far above the longest bounded path (~16 steps)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchService:
    def __init__(self, session: Session, *, settings: Settings, dispatcher: TaskDispatcher) -> None:
        self.session = session
        self.settings = settings
        self.dispatcher = dispatcher

    def create(
        self,
        *,
        question: str,
        analysis_type: AnalysisType | None,
        strategy: RetrievalStrategy | None,
        collection: str | None,
        paper_ids: list[UUID],
    ) -> ResearchRequest:
        filters: dict[str, Any] = {}
        collection_id = None
        if collection:
            found = CollectionRepository(self.session).get_by_name(collection)
            if found is None:
                raise NotFoundError("Collection not found", details={"collection": collection})
            collection_id = found.id
            filters["collection_id"] = str(found.id)
        if paper_ids:
            papers = self.session.scalars(select(Paper).where(Paper.id.in_(paper_ids))).all()
            missing = sorted({str(p) for p in paper_ids} - {str(p.id) for p in papers})
            if missing:
                raise NotFoundError("Papers not found", details={"paper_ids": missing})
            not_ready = [str(p.id) for p in papers if p.status != PaperStatus.READY]
            if not_ready:
                raise ConflictError(
                    "Some papers are not ready yet", details={"paper_ids": not_ready}
                )
            filters["paper_ids"] = [str(p) for p in paper_ids]

        ready_in_scope = (
            select(func.count()).select_from(Paper).where(Paper.status == PaperStatus.READY)
        )
        if collection_id is not None:
            ready_in_scope = ready_in_scope.where(Paper.collection_id == collection_id)
        if paper_ids:
            ready_in_scope = ready_in_scope.where(Paper.id.in_(paper_ids))
        if not self.session.scalar(ready_in_scope):
            raise ConflictError("No ingested (ready) papers match the selected scope")

        strategy = strategy or RetrievalStrategy(self.settings.research_default_strategy)
        request_id = uuid.uuid4()
        request = ResearchRequest(
            id=request_id,
            collection_id=collection_id,
            question=question,
            analysis_type=analysis_type,
            retrieval_strategy=strategy,
            status=JobStatus.QUEUED,
            thread_id=str(request_id),
            options={
                "filters": filters,
                "analysis_type_override": analysis_type.value if analysis_type else None,
            },
        )
        self.session.add(request)
        self.session.commit()
        self._dispatch(request)
        return request

    def _dispatch(self, request: ResearchRequest) -> None:
        try:
            task_id = self.dispatcher.enqueue_research(request.id)
        except DependencyUnavailableError as exc:
            request.status = JobStatus.FAILED
            request.error_message = exc.message
            self.session.commit()
            raise
        request.options = {**request.options, "task_id": task_id}
        self.session.commit()

    def get(self, request_id: UUID) -> ResearchRequest:
        stmt = (
            select(ResearchRequest)
            .where(ResearchRequest.id == request_id)
            .options(selectinload(ResearchRequest.result))
        )
        request = self.session.scalars(stmt).first()
        if request is None:
            raise NotFoundError(
                "Research request not found", details={"request_id": str(request_id)}
            )
        return request

    def resume(self, request_id: UUID) -> ResearchRequest:
        request = self.get(request_id)
        if request.status not in (JobStatus.FAILED, JobStatus.INTERRUPTED):
            raise ConflictError(
                f"Only failed or interrupted requests can be resumed (status: "
                f"{request.status.value})"
            )
        request.status = JobStatus.QUEUED
        request.error_message = None
        self.session.commit()
        self._dispatch(request)
        return request


class ResearchRunner:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        graph: CompiledStateGraph,
        settings: Settings,
    ) -> None:
        self.session_factory = session_factory
        self.graph = graph
        self.settings = settings

    def _set(self, request_id: UUID, **fields: Any) -> None:
        with session_scope(self.session_factory) as s:
            request = s.get(ResearchRequest, request_id)
            if request is not None:
                for key, value in fields.items():
                    setattr(request, key, value)

    def _progress(self, request_id: UUID):  # type: ignore[no-untyped-def]
        def on_node_start(node: str) -> None:
            try:
                self._set(request_id, current_node=node)
            except Exception:  # noqa: BLE001 - progress reporting must never break the run
                logger.warning("Could not record progress", exc_info=True)

        return on_node_start

    def __call__(self, request_id: UUID) -> None:
        self.run(request_id)

    def run(self, request_id: UUID) -> JobStatus | None:
        with session_scope(self.session_factory) as s:
            request = s.get(ResearchRequest, request_id)
            if request is None or request.status == JobStatus.SUCCEEDED:
                return None
            request.status = JobStatus.RUNNING
            request.started_at = request.started_at or _now()
            request.error_message = None
            question, strategy = request.question, request.retrieval_strategy.value
            thread_id = request.thread_id or str(request.id)
            options = dict(request.options or {})

        usage = UsageMetadataCallbackHandler()
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "on_node_start": self._progress(request_id),
                "deadline": time.time() + self.settings.research_timeout_seconds,
            },
            "callbacks": [usage],
            "recursion_limit": RECURSION_LIMIT,
            "run_name": "research_workflow",
            "tags": ["research", strategy],
            "metadata": {
                "request_id": str(request_id),
                "strategy": strategy,
                "collection_id": (options.get("filters") or {}).get("collection_id"),
                "model_provider": self.settings.llm_provider,
                "model_main": self.settings.llm_model,
                "model_fast": self.settings.llm_fast_model,
                "app_version": self.settings.app_version,
            },
        }
        started = time.perf_counter()
        try:
            snapshot = self.graph.get_state(config)
            if snapshot.values and snapshot.next:
                logger.info(
                    "Resuming research run from checkpoint",
                    extra={"request_id": str(request_id), "next": list(snapshot.next)},
                )
                self.graph.invoke(None, config)
            elif not (snapshot.values and snapshot.values.get("final")):
                initial = {
                    "request_id": str(request_id),
                    "question": question,
                    "strategy": strategy,
                    "filters": options.get("filters") or {},
                    "analysis_type_override": options.get("analysis_type_override"),
                }
                self.graph.invoke(initial, config)
            snapshot = self.graph.get_state(config)
        except Exception as exc:
            message = redact(f"{type(exc).__name__}: {exc}")[:1000]
            logger.error(
                "Research run failed", extra={"request_id": str(request_id)}, exc_info=True
            )
            self._set(
                request_id, status=JobStatus.FAILED, error_message=message, finished_at=_now()
            )
            return JobStatus.FAILED

        if snapshot.next:  # paused by interrupt_before (human-in-the-loop)
            self._set(request_id, status=JobStatus.INTERRUPTED, current_node=snapshot.next[0])
            return JobStatus.INTERRUPTED

        self._persist(
            request_id,
            snapshot.values,
            usage.usage_metadata,
            round((time.perf_counter() - started) * 1000, 1),
        )
        return JobStatus.SUCCEEDED

    def _cost(self, usage: dict[str, Any]) -> float | None:
        if not usage or not self.settings.llm_pricing:
            return None
        total = 0.0
        for model, counts in usage.items():
            price = self.settings.llm_pricing.get(model)
            if price is None:
                return None  # unknown price for a used model: do not report a partial cost
            total += counts.get("input_tokens", 0) / 1e6 * price.get("input", 0.0)
            total += counts.get("output_tokens", 0) / 1e6 * price.get("output", 0.0)
        return round(total, 6)

    def _persist(
        self, request_id: UUID, values: dict[str, Any], usage: dict[str, Any], latency_ms: float
    ) -> None:
        final = values.get("final") or {}
        verification = values.get("verification") or {}
        usage_plain = {m: dict(u) for m, u in (usage or {}).items()}
        metrics = {
            "latency_ms": latency_ms,
            "node_timings_ms": values.get("node_timings") or {},
            "retry_count": values.get("retry_count", 0),
            "repair_count": values.get("repair_count", 0),
            "token_usage": usage_plain,
            "cost_usd": self._cost(usage_plain),
            "models": {"main": self.settings.llm_model, "fast": self.settings.llm_fast_model},
            "errors": values.get("errors") or [],
            "trace": values.get("trace") or [],
            "evidence_count": len(values.get("evidence") or []),
        }
        with session_scope(self.session_factory) as s:
            request = s.get(ResearchRequest, request_id)
            if request is None:
                return
            analysis_type = (values.get("analysis") or {}).get("analysis_type")
            if analysis_type:
                request.analysis_type = AnalysisType(analysis_type)
            result = s.scalars(
                select(ResearchResult).where(ResearchResult.request_id == request_id)
            ).first() or ResearchResult(request_id=request_id)
            result.answer = final.get("answer_markdown", "")
            result.findings = {
                "summary": final.get("summary"),
                "sections": final.get("sections", []),
                "uncertainty": final.get("uncertainty"),
                "removed_statements": final.get("removed_statements", []),
                "analysis": values.get("analysis"),
                "queries": final.get("queries", []),
                "assessment": values.get("assessment"),
            }
            result.citations = final.get("references", [])
            result.evidence_sufficient = bool(final.get("evidence_sufficient"))
            result.verification = {
                "metrics": verification.get("metrics"),
                "claims": verification.get("claims", []),
                "llm_checked": verification.get("llm_checked", False),
            }
            result.metrics = metrics
            s.add(result)
            request.status = JobStatus.SUCCEEDED
            request.current_node = None
            request.finished_at = _now()
        logger.info(
            "Research run completed",
            extra={"request_id": str(request_id), "latency_ms": latency_ms},
        )
