"""Celery task entry points. Thin wrappers: all logic lives in the pipeline."""

from __future__ import annotations

import logging
from uuid import UUID

from celery import Task
from celery.signals import worker_process_init, worker_process_shutdown, worker_ready

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.resources import Resources
from app.ingestion.errors import PermanentIngestionError, TransientIngestionError
from app.workers.celery_app import celery_app
from app.workers.dispatch import EVALUATION_TASK_NAME, INGEST_TASK_NAME, RESEARCH_TASK_NAME

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


def _warm_up() -> None:
    """Load models before the first task arrives instead of during it."""
    resources = get_resources()
    _ = resources.dense_embedder.dimension
    if resources.reranker is not None:
        resources.reranker.score("warm up", ["warm up"])


@worker_process_init.connect  # prefork pool: runs in each child process
def _init(**_: object) -> None:
    _warm_up()


@worker_ready.connect  # solo pool (Windows): tasks run in the main process
def _ready(sender: object = None, **_: object) -> None:
    pool_cls = getattr(getattr(sender, "controller", None), "pool_cls", None)
    pool_name = f"{getattr(pool_cls, '__module__', '')}.{getattr(pool_cls, '__name__', '')}"
    if "prefork" not in pool_name:  # prefork children warm up in worker_process_init instead
        _warm_up()


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


@celery_app.task(name=RESEARCH_TASK_NAME, bind=True, max_retries=0)
def run_research(self: Task, request_id: str) -> str | None:
    """Not auto-retried: a failed run is marked failed and resumed explicitly (LLM calls cost
    money). If the worker crashes mid-run, acks_late redelivers the task and the runner continues
    from the last LangGraph checkpoint rather than starting over."""
    status = get_resources().research_runner.run(UUID(request_id))
    return status.value if status else None


@celery_app.task(name=EVALUATION_TASK_NAME, bind=True, max_retries=0)
def run_evaluation(self: Task, batch_id: str) -> None:
    from app.services.evaluations import execute_batch

    execute_batch(get_resources(), UUID(batch_id))
