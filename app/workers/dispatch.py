"""How the API hands ingestion work to an executor.

``CeleryDispatcher`` publishes by task *name*, so the API never imports worker code or loads the
embedding model. ``InlineDispatcher`` runs the pipeline in a background thread of the API
process, for development without Redis and for end-to-end tests.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from app.core.exceptions import DependencyUnavailableError

if TYPE_CHECKING:
    from celery import Celery

    from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

INGEST_TASK_NAME = "researchpilot.ingest_paper"
RESEARCH_TASK_NAME = "researchpilot.run_research"
EVALUATION_TASK_NAME = "researchpilot.run_evaluation"
INGESTION_QUEUE = "ingestion"
RESEARCH_QUEUE = "research"


class TaskDispatcher(Protocol):
    def enqueue_ingestion(self, paper_id: UUID, job_id: UUID) -> str: ...

    def enqueue_research(self, request_id: UUID) -> str: ...

    def enqueue_evaluation(self, batch_id: UUID) -> str: ...

    def close(self) -> None: ...


class CeleryDispatcher:
    def __init__(self, app: Celery) -> None:
        self.app = app

    def _send(self, name: str, args: list[str], queue: str) -> str:
        try:
            result = self.app.send_task(
                name,
                args=args,
                queue=queue,
                retry=True,
                retry_policy={"max_retries": 2, "interval_start": 0.5, "interval_step": 1},
            )
        except Exception as exc:  # kombu raises OperationalError and friends
            raise DependencyUnavailableError("Task queue (Redis) is unavailable") from exc
        return str(result.id)

    def enqueue_ingestion(self, paper_id: UUID, job_id: UUID) -> str:
        return self._send(INGEST_TASK_NAME, [str(paper_id), str(job_id)], INGESTION_QUEUE)

    def enqueue_research(self, request_id: UUID) -> str:
        return self._send(RESEARCH_TASK_NAME, [str(request_id)], RESEARCH_QUEUE)

    def enqueue_evaluation(self, batch_id: UUID) -> str:
        # Shares the research queue: both are long-running, LLM-bound background jobs.
        return self._send(EVALUATION_TASK_NAME, [str(batch_id)], RESEARCH_QUEUE)

    def close(self) -> None:
        self.app.close()


class InlineDispatcher:
    def __init__(
        self,
        pipeline: IngestionPipeline,
        research_runner: Callable[[UUID], object] | None = None,
        max_workers: int = 1,
        evaluation_runner: Callable[[UUID], object] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.research_runner = research_runner
        self.evaluation_runner = evaluation_runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="task")
        self.futures: dict[str, Future[object]] = {}

    def enqueue_research(self, request_id: UUID) -> str:
        if self.research_runner is None:
            raise DependencyUnavailableError("Research execution is not configured")
        task_id = f"inline-{uuid.uuid4().hex}"
        self.futures[task_id] = self._executor.submit(self.research_runner, request_id)
        return task_id

    def enqueue_evaluation(self, batch_id: UUID) -> str:
        if self.evaluation_runner is None:
            raise DependencyUnavailableError("Evaluation execution is not configured")
        task_id = f"inline-{uuid.uuid4().hex}"
        self.futures[task_id] = self._executor.submit(self.evaluation_runner, batch_id)
        return task_id

    def enqueue_ingestion(self, paper_id: UUID, job_id: UUID) -> str:
        task_id = f"inline-{uuid.uuid4().hex}"
        self.futures[task_id] = self._executor.submit(self._run, paper_id, job_id)
        return task_id

    def _run(self, paper_id: UUID, job_id: UUID) -> object:
        try:
            return self.pipeline.run(paper_id, job_id)
        except Exception:  # already recorded on the job; keep the thread pool healthy
            logger.info("Inline ingestion ended with an error", extra={"paper_id": str(paper_id)})
            return None

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
