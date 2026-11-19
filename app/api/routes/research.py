"""Research endpoints. A query is accepted immediately (202) and executed in the background;
clients poll ``GET /research/{request_id}`` (``current_node`` shows progress)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_resources
from app.api.schemas.research import ResearchAccepted, ResearchQuery, ResearchStatus
from app.core.exceptions import ErrorResponse
from app.core.resources import Resources
from app.services.research import ResearchService

router = APIRouter(prefix="/research", tags=["research"])


def get_research_service(
    db: Session = Depends(get_db), resources: Resources = Depends(get_resources)
) -> ResearchService:
    return ResearchService(db, settings=resources.settings, dispatcher=resources.dispatcher)


def _accepted(request: Request, request_id: UUID, job_status) -> ResearchAccepted:  # type: ignore[no-untyped-def]
    return ResearchAccepted(
        request_id=request_id,
        status=job_status,
        status_url=str(request.url_for("get_research", request_id=str(request_id)).path),
    )


@router.post(
    "/query",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ResearchAccepted,
    summary="Start a research question (runs in the background)",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def submit_query(
    body: ResearchQuery, request: Request, service: ResearchService = Depends(get_research_service)
) -> ResearchAccepted:
    created = service.create(
        question=body.question,
        analysis_type=body.analysis_type,
        strategy=body.strategy,
        collection=body.collection,
        paper_ids=body.paper_ids,
    )
    return _accepted(request, created.id, created.status)


@router.get(
    "/{request_id}",
    name="get_research",
    response_model=ResearchStatus,
    summary="Status, progress and (when finished) the answer of a research request",
    responses={404: {"model": ErrorResponse}},
)
def get_research(
    request_id: UUID, service: ResearchService = Depends(get_research_service)
) -> ResearchStatus:
    return ResearchStatus.from_request(service.get(request_id))


@router.post(
    "/{request_id}/resume",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ResearchAccepted,
    summary="Resume a failed or interrupted run from its last checkpoint",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def resume_research(
    request_id: UUID, request: Request, service: ResearchService = Depends(get_research_service)
) -> ResearchAccepted:
    resumed = service.resume(request_id)
    return _accepted(request, resumed.id, resumed.status)
