"""How the API hands ingestion work to an executor.

``CeleryDispatcher`` publishes by task *name*, so the API never imports worker code or loads the
embedding model. ``InlineDispatcher`` runs the pipeline in a background thread of the API
process, for development without Redis and for end-to-end tests.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from app.core.exceptions import DependencyUnavailableError

if TYPE_CHECKING:
    from celery import Celery

    from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

INGEST_TASK_NAME = "researchpilot.ingest_paper"
INGESTION_QUEUE = "ingestion"


class TaskDispatcher(Protocol):
    def enqueue_ingestion(self, paper_id: UUID, job_id: UUID) -> str: ...

    def close(self) -> None: ...


class CeleryDispatcher:
    def __init__(self, app: Celery) -> None:
        self.app = app

    def enqueue_ingestion(self, paper_id: UUID, job_id: UUID) -> str:
        try:
            result = self.app.send_task(
                INGEST_TASK_NAME,
                args=[str(paper_id), str(job_id)],
                queue=INGESTION_QUEUE,
                retry=True,
                retry_policy={"max_retries": 2, "interval_start": 0.5, "interval_step": 1},
            )
        except Exception as exc:  # kombu raises OperationalError and friends
            raise DependencyUnavailableError("Task queue (Redis) is unavailable") from exc
        return str(result.id)

    def close(self) -> None:
        self.app.close()


class InlineDispatcher:
    def __init__(self, pipeline: IngestionPipeline, max_workers: int = 1) -> None:
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")
        self.futures: dict[str, Future[object]] = {}

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
