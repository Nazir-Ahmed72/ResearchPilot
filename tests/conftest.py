"""Shared fixtures.

Tests never need Docker or network access: they use a temporary SQLite database, Qdrant's
in-memory mode, and stubbed Redis/health checks. Tests marked ``postgres`` run only when
``TEST_DATABASE_URL`` points at a real PostgreSQL instance.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Make sure a developer's local .env never leaks into tests.
os.environ.setdefault("ENVIRONMENT", "test")

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.resources import Resources  # noqa: E402
from app.database.models import Base  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import FakeDenseEmbedder, RecordingDispatcher  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    get_settings.cache_clear()
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        qdrant_location=":memory:",
        redis_url="redis://127.0.0.1:1/0",  # deliberately unreachable
        upload_dir=tmp_path / "uploads",
        hf_home=tmp_path / "hf",
        temp_dir=None,
        task_backend="inline",
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        max_upload_size_mb=2,
        max_files_per_upload=3,
        log_json=True,
        health_check_timeout_seconds=1.0,
    )


@pytest.fixture
def embedder() -> FakeDenseEmbedder:
    return FakeDenseEmbedder()


@pytest.fixture
def dispatcher() -> RecordingDispatcher:
    return RecordingDispatcher()


@pytest.fixture
def resources(
    settings: Settings, embedder: FakeDenseEmbedder, dispatcher: RecordingDispatcher
) -> Iterator[Resources]:
    res = Resources.from_settings(settings, dense_embedder=embedder, dispatcher=dispatcher)
    Base.metadata.create_all(res.engine)
    yield res
    res.close()


@pytest.fixture
def app(resources: Resources) -> FastAPI:
    return create_app(resources=resources)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
