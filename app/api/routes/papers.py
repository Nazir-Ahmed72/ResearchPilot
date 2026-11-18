"""Paper endpoints.

Handlers are plain ``def``: FastAPI runs them in a worker thread, so synchronous database and
Qdrant calls never block the event loop.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_resources
from app.api.schemas.papers import (
    PaperDetail,
    PaperList,
    PaperSummary,
    ReingestResponse,
    UploadResponse,
    UploadResult,
)
from app.core.exceptions import ErrorResponse
from app.core.resources import Resources
from app.database.models import PaperStatus
from app.services.papers import IncomingFile, PaperService

router = APIRouter(prefix="/papers", tags=["papers"])

_READ_CHUNK = 1024 * 1024


def get_paper_service(
    db: Session = Depends(get_db), resources: Resources = Depends(get_resources)
) -> PaperService:
    return PaperService(
        db,
        settings=resources.settings,
        storage=resources.storage,
        store=resources.chunk_store,
        dispatcher=resources.dispatcher,
    )


def _read_limited(upload: UploadFile, limit: int) -> IncomingFile:
    """Read at most ``limit + 1`` bytes so an oversized file is detected without loading it all."""
    buffer = bytearray()
    while len(buffer) <= limit:
        chunk = upload.file.read(_READ_CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
    truncated = len(buffer) > limit
    return IncomingFile(
        filename=upload.filename,
        content_type=upload.content_type,
        data=b"" if truncated else bytes(buffer),
        truncated=truncated,
    )


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    summary="Upload one or more PDF papers for background ingestion",
    responses={
        400: {"model": ErrorResponse, "description": "No file accepted (invalid PDFs)"},
        409: {"model": ErrorResponse, "description": "All files were duplicates"},
        413: {"model": ErrorResponse, "description": "All files exceeded the size limit"},
        415: {"model": ErrorResponse, "description": "All files were not PDFs"},
    },
)
def upload_papers(
    files: list[UploadFile] = File(..., description="PDF files"),
    collection: str | None = Form(None, description="Optional collection name (created if new)"),
    service: PaperService = Depends(get_paper_service),
    resources: Resources = Depends(get_resources),
) -> UploadResponse:
    limit = resources.settings.max_upload_size_bytes
    incoming = [_read_limited(f, limit) for f in files]
    outcomes = service.upload(incoming, collection)
    return UploadResponse(
        accepted=sum(o.status == "accepted" for o in outcomes),
        results=[
            UploadResult(
                filename=o.filename,
                status=o.status,
                paper_id=o.paper_id,
                job_id=o.job_id,
                error_code=o.error_code,
                message=o.message,
            )
            for o in outcomes
        ],
    )


@router.get("", response_model=PaperList, summary="List papers")
def list_papers(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: PaperStatus | None = Query(None, alias="status"),
    collection: str | None = Query(None, max_length=200),
    service: PaperService = Depends(get_paper_service),
) -> PaperList:
    items, total = service.list(
        limit=limit, offset=offset, status=status_filter, collection=collection
    )
    return PaperList(
        items=[PaperSummary.from_paper(p) for p in items], total=total, limit=limit, offset=offset
    )


@router.get(
    "/{paper_id}",
    response_model=PaperDetail,
    summary="Get a paper with its ingestion history",
    responses={404: {"model": ErrorResponse}},
)
def get_paper(paper_id: UUID, service: PaperService = Depends(get_paper_service)) -> PaperDetail:
    return PaperDetail.from_paper(service.get(paper_id))


@router.delete(
    "/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a paper, its vectors and its stored file",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def delete_paper(paper_id: UUID, service: PaperService = Depends(get_paper_service)) -> Response:
    service.delete(paper_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{paper_id}/reingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReingestResponse,
    summary="Re-run ingestion (e.g. after a failure or a chunking change)",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def reingest_paper(
    paper_id: UUID, service: PaperService = Depends(get_paper_service)
) -> ReingestResponse:
    job = service.reingest(paper_id)
    return ReingestResponse(paper_id=paper_id, job_id=job.id, status=job.status)
