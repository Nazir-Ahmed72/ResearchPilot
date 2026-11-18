"""Paper use cases: upload, list, get, delete, re-ingest.

Consistency between PostgreSQL, Qdrant and file storage:

* Upload: the file is written first, then the DB row. If the DB insert fails, the file is
  removed. If queueing fails, the row stays, marked ``failed``, so it can be re-ingested.
* Delete: the paper is marked ``deleting`` and committed, then vectors and file are removed,
  then the row. If vector deletion fails, the row stays ``deleting`` and the delete can simply
  be repeated, so vectors are never orphaned. The worker checks for ``deleting`` and backs off.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    ConflictError,
    DependencyUnavailableError,
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.database.models import IngestionJob, JobStatus, Paper, PaperStatus
from app.database.repositories.papers import (
    CollectionRepository,
    IngestionJobRepository,
    PaperRepository,
)
from app.ingestion.errors import PdfValidationError
from app.ingestion.storage import LocalFileStorage
from app.ingestion.validation import validate_pdf
from app.vectorstore.qdrant_store import QdrantChunkStore
from app.workers.dispatch import TaskDispatcher

logger = logging.getLogger(__name__)

UploadStatus = Literal["accepted", "duplicate", "rejected", "failed"]
_ERROR_STATUS = {
    "payload_too_large": PayloadTooLargeError,
    "unsupported_media_type": UnsupportedMediaTypeError,
    "invalid_pdf": InvalidInputError,
    "duplicate": ConflictError,
    "dependency_unavailable": DependencyUnavailableError,
}


@dataclass
class IncomingFile:
    filename: str | None
    content_type: str | None
    data: bytes
    truncated: bool = False  # read stopped at the size limit


@dataclass
class UploadOutcome:
    filename: str
    status: UploadStatus
    paper_id: UUID | None = None
    job_id: UUID | None = None
    error_code: str | None = None
    message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class UploadRejectedError(AppError):
    """No file in the request was accepted; status mirrors the (shared) reason."""

    def __init__(self, outcomes: list[UploadOutcome]) -> None:
        codes = {o.error_code for o in outcomes}
        error_cls = (
            _ERROR_STATUS.get(codes.pop(), InvalidInputError)
            if len(codes) == 1
            else InvalidInputError
        )
        self.status_code = error_cls.status_code
        self.code = "upload_rejected"
        super().__init__(
            "No files were accepted",
            details=[
                {
                    "filename": o.filename,
                    "error_code": o.error_code,
                    "message": o.message,
                    "paper_id": o.paper_id,
                }
                for o in outcomes
            ],
        )


class PaperService:
    def __init__(
        self,
        session: Session,
        *,
        settings: Settings,
        storage: LocalFileStorage,
        store: QdrantChunkStore,
        dispatcher: TaskDispatcher,
    ) -> None:
        self.session = session
        self.settings = settings
        self.storage = storage
        self.store = store
        self.dispatcher = dispatcher
        self.papers = PaperRepository(session)
        self.collections = CollectionRepository(session)
        self.jobs = IngestionJobRepository(session)

    # -- upload --------------------------------------------------------------------------------

    def upload(self, files: list[IncomingFile], collection_name: str | None) -> list[UploadOutcome]:
        if not files:
            raise InvalidInputError("At least one file is required")
        if len(files) > self.settings.max_files_per_upload:
            raise InvalidInputError(
                f"At most {self.settings.max_files_per_upload} files per request"
            )
        collection_id = None
        if collection_name:
            name = collection_name.strip()
            if not name or len(name) > 200:
                raise InvalidInputError("Collection name must be 1-200 characters")
            collection_id = self.collections.get_or_create(name).id
            self.session.commit()

        seen_hashes: dict[str, UUID] = {}
        outcomes = [self._upload_one(f, collection_id, seen_hashes) for f in files]
        if not any(o.status == "accepted" for o in outcomes):
            raise UploadRejectedError(outcomes)
        return outcomes

    def _upload_one(
        self, file: IncomingFile, collection_id: UUID | None, seen: dict[str, UUID]
    ) -> UploadOutcome:
        name = file.filename or "upload.pdf"
        try:
            if file.truncated:
                raise PdfValidationError(
                    "payload_too_large",
                    f"File exceeds the {self.settings.max_upload_size_mb} MB limit",
                )
            validated = validate_pdf(
                file.data,
                filename=file.filename,
                content_type=file.content_type,
                max_size_bytes=self.settings.max_upload_size_bytes,
                max_pages=self.settings.max_pdf_pages,
            )
        except PdfValidationError as exc:
            return UploadOutcome(name, "rejected", error_code=exc.code, message=exc.message)
        name = validated.filename

        content_hash = hashlib.sha256(file.data).hexdigest()
        existing_id = seen.get(content_hash)
        if existing_id is None:
            existing = self.papers.get_by_hash(content_hash)
            existing_id = existing.id if existing else None
        if existing_id is not None:
            return UploadOutcome(
                name,
                "duplicate",
                paper_id=existing_id,
                error_code="duplicate",
                message="This PDF has already been uploaded",
            )

        paper_id = uuid.uuid4()
        key = self.storage.save(paper_id, file.data)
        paper = Paper(
            id=paper_id,
            collection_id=collection_id,
            content_hash=content_hash,
            original_filename=name,
            storage_path=key,
            file_size_bytes=validated.size_bytes,
            page_count=validated.page_count,
            status=PaperStatus.PENDING,
        )
        job = IngestionJob(id=uuid.uuid4(), paper=paper, status=JobStatus.QUEUED)
        self.papers.add(paper)
        self.jobs.add(job)  # explicit: SQLAlchemy 2.0 does not cascade via backrefs
        try:
            self.session.commit()
        except IntegrityError:
            # A concurrent request inserted the same content first.
            self.session.rollback()
            self.storage.delete(key)
            winner = self.papers.get_by_hash(content_hash)
            return UploadOutcome(
                name,
                "duplicate",
                paper_id=winner.id if winner else None,
                error_code="duplicate",
                message="This PDF has already been uploaded",
            )
        seen[content_hash] = paper_id

        try:
            job.task_id = self.dispatcher.enqueue_ingestion(paper_id, job.id)
            self.session.commit()
        except DependencyUnavailableError as exc:
            self.session.rollback()
            self._mark_enqueue_failed(paper_id, job.id, exc.message)
            return UploadOutcome(
                name,
                "failed",
                paper_id=paper_id,
                job_id=job.id,
                error_code=exc.code,
                message=f"{exc.message}; retry with POST /papers/{paper_id}/reingest",
            )
        logger.info(
            "Paper accepted", extra={"paper_id": str(paper_id), "pages": validated.page_count}
        )
        return UploadOutcome(name, "accepted", paper_id=paper_id, job_id=job.id)

    def _mark_enqueue_failed(self, paper_id: UUID, job_id: UUID, message: str) -> None:
        paper = self.papers.get(paper_id)
        job = self.session.get(IngestionJob, job_id)
        if paper is not None:
            paper.status = PaperStatus.FAILED
            paper.error_message = message
        if job is not None:
            job.status = JobStatus.FAILED
            job.error_message = message
            job.finished_at = datetime.now(timezone.utc)
        self.session.commit()

    # -- queries -------------------------------------------------------------------------------

    def get(self, paper_id: UUID) -> Paper:
        paper = self.papers.get_with_jobs(paper_id)
        if paper is None:
            raise NotFoundError("Paper not found", details={"paper_id": str(paper_id)})
        return paper

    def list(
        self, *, limit: int, offset: int, status: PaperStatus | None, collection: str | None
    ) -> tuple[list[Paper], int]:
        collection_id = None
        if collection:
            found = self.collections.get_by_name(collection)
            if found is None:
                return [], 0
            collection_id = found.id
        items, total = self.papers.list(
            limit=limit, offset=offset, status=status, collection_id=collection_id
        )
        return list(items), total

    # -- delete / re-ingest --------------------------------------------------------------------

    def delete(self, paper_id: UUID) -> None:
        paper = self.get(paper_id)
        if paper.status == PaperStatus.PROCESSING:
            raise ConflictError(
                "Paper is being processed; retry after ingestion finishes",
                details={"paper_id": str(paper_id), "status": paper.status.value},
            )
        paper.status = PaperStatus.DELETING
        self.session.commit()
        try:
            self.store.delete_paper(paper_id)
        except Exception as exc:
            logger.error("Vector deletion failed", extra={"paper_id": str(paper_id)}, exc_info=True)
            raise DependencyUnavailableError(
                "Could not delete the paper's vectors; the paper is marked 'deleting' and the "
                "delete can be retried"
            ) from exc
        self.storage.delete(paper.storage_path)
        self.papers.delete(paper)
        self.session.commit()
        logger.info("Paper deleted", extra={"paper_id": str(paper_id)})

    def reingest(self, paper_id: UUID) -> IngestionJob:
        paper = self.get(paper_id)
        if paper.status not in (PaperStatus.READY, PaperStatus.FAILED):
            raise ConflictError(
                f"Paper cannot be re-ingested while {paper.status.value}",
                details={"paper_id": str(paper_id), "status": paper.status.value},
            )
        if not self.storage.exists(paper.storage_path):
            raise ConflictError("The original PDF is no longer stored; upload it again")
        job = self.jobs.add(IngestionJob(id=uuid.uuid4(), paper=paper, status=JobStatus.QUEUED))
        paper.status = PaperStatus.PENDING
        paper.error_message = None
        self.session.commit()
        try:
            job.task_id = self.dispatcher.enqueue_ingestion(paper.id, job.id)
            self.session.commit()
        except DependencyUnavailableError:
            self.session.rollback()
            self._mark_enqueue_failed(paper.id, job.id, "Task queue (Redis) is unavailable")
            raise
        return job
