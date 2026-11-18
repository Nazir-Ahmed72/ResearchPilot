from __future__ import annotations

import ssl

from app.core.config import Settings
from app.workers.celery_config import create_celery_app
from app.workers.dispatch import INGEST_TASK_NAME, INGESTION_QUEUE


def _app(url: str):  # type: ignore[no-untyped-def]
    return create_celery_app(Settings(_env_file=None, redis_url=url))  # type: ignore[call-arg]


def test_tls_broker_requires_certificate_verification() -> None:
    app = _app("rediss://default:pw@example.upstash.io:6379")
    assert app.conf.broker_use_ssl == {"ssl_cert_reqs": ssl.CERT_REQUIRED}


def test_plain_redis_has_no_tls_options() -> None:
    assert not _app("redis://localhost:6379/0").conf.broker_use_ssl


def test_reliability_and_low_redis_traffic_settings() -> None:
    conf = _app("redis://localhost:6379/0").conf
    assert conf.task_acks_late and conf.task_reject_on_worker_lost
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_ignore_result and conf.result_backend is None
    assert conf.worker_enable_remote_control is False
    opts = conf.broker_transport_options
    assert opts["polling_interval"] == 10.0
    assert opts["visibility_timeout"] > conf.task_time_limit
    assert opts["socket_timeout"] > opts["polling_interval"]
    assert conf.task_default_queue == INGESTION_QUEUE


def test_worker_registers_the_ingest_task(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    from app.workers import tasks

    assert INGEST_TASK_NAME in tasks.celery_app.tasks
    assert tasks.ingest_paper.max_retries >= 0
