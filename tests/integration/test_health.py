"""Health endpoint against real SQLite + in-memory Qdrant; Redis is intentionally unreachable."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_liveness_has_no_dependencies(client: TestClient) -> None:
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_reports_real_checks(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    body = resp.json()
    assert body["checks"]["database"]["status"] == "ok"
    assert body["checks"]["qdrant"]["status"] == "ok"
    # No Redis is running in tests, so readiness must honestly report degraded + 503.
    assert body["checks"]["redis"]["status"] == "error"
    assert body["status"] == "degraded"
    assert resp.status_code == 503
    assert body["version"] == "0.1.0"


def test_readiness_ok_when_all_checks_pass(app: FastAPI) -> None:
    app.state.health_checks = {"database": lambda: None, "redis": lambda: None}
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_hanging_dependency_times_out_instead_of_blocking(app: FastAPI) -> None:
    app.state.health_checks = {"slow": lambda: time.sleep(3)}
    with TestClient(app) as client:
        start = time.perf_counter()
        resp = client.get("/api/v1/health")
        elapsed = time.perf_counter() - start
    assert resp.status_code == 503
    assert resp.json()["checks"]["slow"]["error"].startswith("timed out")
    assert elapsed < 2.5


def test_failure_details_do_not_leak_exception_messages(app: FastAPI) -> None:
    def failing() -> None:
        raise ConnectionError("could not connect to postgresql://u:hunter2@db")

    app.state.health_checks = {"database": failing}
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
    assert resp.json()["checks"]["database"]["error"] == "ConnectionError"
    assert "hunter2" not in resp.text


def test_openapi_docs_are_served(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/api/v1/health" in resp.json()["paths"]
