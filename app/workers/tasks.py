"""Celery task entry points. Thin wrappers: all logic lives in the pipeline."""

from __future__ import annotations

import logging
from uuid import UUID

from celery import Task
from celery.signals import worker_process_init, worker_process_shutdown

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.resources import Resources
from app.ingestion.errors import PermanentIngestionError, TransientIngestionError
from app.workers.celery_app import celery_app
from app.workers.dispatch import INGEST_TASK_NAME

logger = logging.getLogger(__name__)
_resources: Resources | None = None


def get_resources() -> Resources:
    """One set of clients and one loaded model per worker process."""
    global _resources
    if _resources is None:
        settings = get_settings()
        configure_logging(settings.log_level, settings.log_json)
        _resources = Resources.from_settings(settings)
    return _resources


@worker_process_init.connect
def _init(**_: object) -> None:
    resources = get_resources()
    _ = resources.dense_embedder.dimension  # load the model before the first task arrives


@worker_process_shutdown.connect
def _shutdown(**_: object) -> None:
    if _resources is not None:
        _resources.close()


@celery_app.task(
    name=INGEST_TASK_NAME,
    bind=True,
    autoretry_for=(TransientIngestionError,),
    retry_backoff=5,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=get_settings().ingestion_max_retries,
)
def ingest_paper(self: Task, paper_id: str, job_id: str) -> dict[str, object] | None:
    final_attempt = self.request.retries >= (self.max_retries or 0)
    try:
        result = get_resources().ingestion_pipeline.run(
            UUID(paper_id), UUID(job_id), final_attempt=final_attempt
        )
    except PermanentIngestionError:
        return None  # bad input: already recorded as failed on the paper/job; never retry
    if result is None:
        return None
    return {"paper_id": paper_id, "chunks": result.chunk_count, "pages": result.page_count}
