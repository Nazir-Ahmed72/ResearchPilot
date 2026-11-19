"""Evaluation endpoints: start an evaluation batch in the background and read results."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_resources
from app.api.schemas.evaluations import (
    EvaluationAccepted,
    EvaluationResults,
    EvaluationRunOut,
    EvaluationRunRequest,
)
from app.core.exceptions import ErrorResponse
from app.core.resources import Resources
from app.evaluation.runner import EvaluationOptions
from app.services.evaluations import EvaluationService

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def get_evaluation_service(
    db: Session = Depends(get_db), resources: Resources = Depends(get_resources)
) -> EvaluationService:
    return EvaluationService(db, resources.dispatcher)


@router.post(
    "/run",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=EvaluationAccepted,
    summary="Run retrieval experiments (E1-E5) and/or the end-to-end evaluation in the background",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def run_evaluation(
    body: EvaluationRunRequest, service: EvaluationService = Depends(get_evaluation_service)
) -> EvaluationAccepted:
    options = EvaluationOptions(
        dataset=body.dataset,
        experiments=list(dict.fromkeys(body.experiments)),
        top_k=body.top_k,
        include_generation=body.include_generation,
        generation_strategy=body.generation_strategy.value,
        question_ids=body.question_ids,
        limit=body.limit,
        ragas=body.ragas,
        pace_seconds=body.pace_seconds,
        collection=body.collection,
        langsmith=body.langsmith,
        name=body.name,
    )
    batch_id, runs = service.create(options)
    return EvaluationAccepted(
        batch_id=batch_id, runs=[EvaluationRunOut.model_validate(r) for r in runs]
    )


@router.get("/results", response_model=EvaluationResults, summary="List evaluation runs")
def list_results(
    limit: int = Query(20, ge=1, le=200),
    experiment: str | None = Query(None, max_length=10),
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResults:
    return EvaluationResults(
        items=[
            EvaluationRunOut.model_validate(r)
            for r in service.list(limit=limit, experiment=experiment)
        ]
    )


@router.get(
    "/results/{run_id}",
    response_model=EvaluationRunOut,
    summary="One evaluation run with its metrics",
    responses={404: {"model": ErrorResponse}},
)
def get_result(
    run_id: UUID, service: EvaluationService = Depends(get_evaluation_service)
) -> EvaluationRunOut:
    return EvaluationRunOut.model_validate(service.get(run_id))
