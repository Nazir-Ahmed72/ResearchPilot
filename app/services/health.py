"""Dependency health checks.

Each check is a small synchronous function that raises on failure. Checks run concurrently in
worker threads with a hard timeout, so one hanging dependency cannot stall the health endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import text

from app.core.resources import Resources

logger = logging.getLogger(__name__)

CheckFn = Callable[[], None]
CheckStatus = Literal["ok", "error"]


class CheckResult(BaseModel):
    status: CheckStatus
    latency_ms: float
    error: str | None = None


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    checks: dict[str, CheckResult]


def default_checks(resources: Resources) -> dict[str, CheckFn]:
    def database() -> None:
        with resources.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def redis() -> None:
        resources.redis.ping()

    def qdrant() -> None:
        resources.qdrant.get_collections()

    return {"database": database, "redis": redis, "qdrant": qdrant}


async def _run_check(name: str, fn: CheckFn, timeout: float) -> CheckResult:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
        status: CheckStatus = "ok"
        error = None
    except asyncio.TimeoutError:
        status, error = "error", f"timed out after {timeout:.1f}s"
    except Exception as exc:  # noqa: BLE001 - any failure means the dependency is unhealthy
        # Only the exception type is exposed; messages can contain hostnames or credentials.
        status, error = "error", type(exc).__name__
        logger.warning("Health check failed", extra={"check": name, "error_type": error})
    return CheckResult(
        status=status, latency_ms=round((time.perf_counter() - start) * 1000, 2), error=error
    )


async def run_health_checks(
    checks: Mapping[str, CheckFn], *, version: str, environment: str, timeout: float
) -> HealthReport:
    names = list(checks)
    results = await asyncio.gather(*(_run_check(n, checks[n], timeout) for n in names))
    by_name = dict(zip(names, results, strict=True))
    overall = "ok" if all(r.status == "ok" for r in results) else "degraded"
    return HealthReport(status=overall, version=version, environment=environment, checks=by_name)
