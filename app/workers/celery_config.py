"""Celery application.

Start a worker (Windows needs the single-process ``solo`` pool)::

    celery -A app.workers.celery_app worker -Q ingestion,research --pool=solo --loglevel=INFO \
        --without-gossip --without-mingle --without-heartbeat

The flags and settings below minimise idle Redis traffic, which matters on Upstash's metered
free tier: no result backend (status lives in PostgreSQL), no remote control or event
broadcasting, and a long blocking poll interval.
"""

from __future__ import annotations

import ssl

from celery import Celery

from app.core.config import Settings
from app.workers.dispatch import INGESTION_QUEUE, RESEARCH_QUEUE


def create_celery_app(settings: Settings) -> Celery:
    app = Celery("researchpilot", broker=settings.redis_url, include=["app.workers.tasks"])
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        task_ignore_result=True,
        result_backend=None,
        task_default_queue=INGESTION_QUEUE,
        task_routes={
            "researchpilot.ingest_paper": {"queue": INGESTION_QUEUE},
            "researchpilot.run_research": {"queue": RESEARCH_QUEUE},
            "researchpilot.run_evaluation": {"queue": RESEARCH_QUEUE},
        },
        # Reliability: acknowledge only after the task finishes, so a crashed worker's task is
        # redelivered. Safe because the pipeline is idempotent.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=30 * 60,
        task_soft_time_limit=25 * 60,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": 60 * 60,  # > task_time_limit, or tasks get redelivered early
            "polling_interval": settings.celery_broker_polling_interval,
            "health_check_interval": 60,
            "socket_timeout": settings.celery_broker_polling_interval + 20,
            "socket_connect_timeout": 10,
        },
        worker_enable_remote_control=False,
        worker_send_task_events=False,
        task_send_sent_event=False,
        worker_hijack_root_logger=False,
    )
    if settings.redis_url.startswith("rediss://"):
        app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    return app
