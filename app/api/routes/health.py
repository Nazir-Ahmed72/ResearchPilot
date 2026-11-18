from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel

from app.api.dependencies import get_resources
from app.core.resources import Resources
from app.services.health import HealthReport, default_checks, run_health_checks

router = APIRouter(prefix="/health", tags=["health"])


class LivenessResponse(BaseModel):
    status: str = "ok"


@router.get(
    "",
    response_model=HealthReport,
    summary="Readiness: checks PostgreSQL, Redis and Qdrant",
    responses={503: {"model": HealthReport, "description": "One or more dependencies failed"}},
)
async def readiness(
    request: Request, response: Response, resources: Resources = Depends(get_resources)
) -> HealthReport:
    checks = getattr(request.app.state, "health_checks", None) or default_checks(resources)
    report = await run_health_checks(
        checks,
        version=resources.settings.app_version,
        environment=resources.settings.environment,
        timeout=resources.settings.health_check_timeout_seconds,
    )
    if report.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness: the process is up (no dependency checks)",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse()
