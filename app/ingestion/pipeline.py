"""The ingestion pipeline: stored PDF -> chunks with provenance -> Qdrant, status in PostgreSQL.

Runs in the worker (or in-process for the ``inline`` backend). It is independent of the
query-time research workflow and safe to re-run:

* Chunk IDs are deterministic, and existing vectors for the paper are deleted before upsert,
  so a retry or re-ingestion never duplicates chunks.
* Database state is committed in short transactions (start, finish) so the API can show
  progress, and no DB transaction is held open during slow embedding work.
* On failure, vectors written so far are removed (compensating action) and the paper and job
  are marked ``failed`` with a sanitised message. Transient errors are re-raised so Celery can
  retry them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.core.logging import redact
from app.database.models import IngestionJob, JobStatus, Paper, PaperStatus
from app.database.session import session_scope
from app.embeddings.dense import DenseEmbedder
from app.embeddings.sparse import Bm25SparseEncoder
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.errors import PermanentIngestionError, TransientIngestionError
from app.ingestion.pdf_loader import ExtractedDocument, load_pdf
from app.ingestion.storage import LocalFileStorage
from app.ingestion.text_cleaner import clean_blocks
from app.vectorstore.qdrant_store import ChunkPoint, QdrantChunkStore

logger = logging.getLogger(__name__)

# Tokens reserved for the "title / section" context header prepended before embedding.
CONTEXT_HEADER_TOKENS = 48


@dataclass
class IngestionConfig:
    chunk_size: int
    chunk_overlap: int
    include_references: bool = False
    upsert_batch_size: int = 64


@dataclass
class IngestionResult:
    paper_id: UUID
    page_count: int
    chunk_count: int
    timings_ms: dict[str, float] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def embedding_text(title: str | None, section: str, text: str) -> str:
    """Contextual header: chunks like 'Results ... improved by 4%' become self-describing."""
    header = " | ".join(part for part in (title, section) if part)
    return f"{header}\n{text}" if header else text


def chunk_payload(paper: Paper, doc: ExtractedDocument, chunk: Chunk) -> dict[str, Any]:
    meta = doc.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "paper_id": str(paper.id),
        "collection_id": str(paper.collection_id) if paper.collection_id else None,
        "text": chunk.text,
        "section": chunk.section,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "token_count": chunk.token_count,
        "title": meta.title or paper.title,
        "authors": meta.authors or paper.authors,
        "year": meta.year or paper.publication_year,
        "doi": meta.doi or paper.doi,
        "source": paper.original_filename,
    }


class IngestionPipeline:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        storage: LocalFileStorage,
        store: QdrantChunkStore,
        dense: DenseEmbedder,
        sparse: Bm25SparseEncoder,
        config: IngestionConfig,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.config = config

    # -- state transitions ---------------------------------------------------------------------

    def _start(self, paper_id: UUID, job_id: UUID) -> tuple[str, str] | None:
        with session_scope(self.session_factory) as s:
            paper = s.get(Paper, paper_id)
            job = s.get(IngestionJob, job_id)
            if paper is None or job is None or paper.status == PaperStatus.DELETING:
                reason = (
                    "paper not found"
                    if paper is None
                    else "job not found"
                    if job is None
                    else "paper is being deleted"
                )
                logger.info(
                    "Skipping ingestion", extra={"paper_id": str(paper_id), "reason": reason}
                )
                return None
            job.status = JobStatus.RUNNING
            job.attempts += 1
            job.started_at = _now()
            job.error_message = None
            paper.status = PaperStatus.PROCESSING
            paper.error_message = None
            return paper.storage_path, paper.content_hash

    def _finish(
        self,
        paper_id: UUID,
        job_id: UUID,
        doc: ExtractedDocument,
        chunks: list[Chunk],
        timings: dict[str, float],
    ) -> bool:
        with session_scope(self.session_factory) as s:
            paper = s.get(Paper, paper_id)
            job = s.get(IngestionJob, job_id)
            if paper is None or job is None or paper.status == PaperStatus.DELETING:
                return False  # deleted while we were working; caller cleans up vectors
            meta = doc.metadata
            # Metadata is extraction-owned (not user-editable), so a re-ingestion with improved
            # extraction refreshes it; previous values are kept only when nothing was found.
            paper.title = meta.title or paper.title
            paper.authors = meta.authors or paper.authors
            paper.publication_year = meta.year or paper.publication_year
            paper.doi = meta.doi or paper.doi
            paper.page_count = doc.page_count
            paper.chunk_count = len(chunks)
            paper.pdf_metadata = {
                **meta.raw,
                **({"arxiv_id": meta.arxiv_id} if meta.arxiv_id else {}),
            }
            paper.status = PaperStatus.READY
            job.status = JobStatus.SUCCEEDED
            job.finished_at = _now()
            job.stats = {
                "page_count": doc.page_count,
                "chunk_count": len(chunks),
                "sections": sorted({c.section for c in chunks}),
                "timings_ms": timings,
            }
            return True

    def _fail(self, paper_id: UUID, job_id: UUID, exc: BaseException, *, final: bool) -> None:
        message = redact(f"{type(exc).__name__}: {exc}")[:1000]
        with session_scope(self.session_factory) as s:
            paper = s.get(Paper, paper_id)
            job = s.get(IngestionJob, job_id)
            if job is not None:
                job.status = JobStatus.FAILED if final else JobStatus.QUEUED
                job.error_message = message
                job.finished_at = _now() if final else None
            if paper is not None and paper.status != PaperStatus.DELETING:
                paper.status = PaperStatus.FAILED if final else PaperStatus.PENDING
                paper.error_message = message

    # -- main entry point ----------------------------------------------------------------------

    def run(
        self, paper_id: UUID, job_id: UUID, *, final_attempt: bool = True
    ) -> IngestionResult | None:
        started = self._start(paper_id, job_id)
        if started is None:
            return None
        storage_key, content_hash = started
        log_ctx = {"paper_id": str(paper_id), "job_id": str(job_id)}
        logger.info("Ingestion started", extra=log_ctx)
        timings: dict[str, float] = {}
        vectors_written = False
        try:
            t = time.perf_counter()
            data = self.storage.read(storage_key)
            doc = load_pdf(data)
            blocks = clean_blocks(doc.blocks, doc.page_count)
            timings["extract"] = _ms(t)

            t = time.perf_counter()
            chunk_size = min(self.config.chunk_size, self.dense.max_tokens - CONTEXT_HEADER_TOKENS)
            chunks = chunk_document(
                blocks,
                document_key=content_hash,
                chunk_size=chunk_size,
                overlap=min(self.config.chunk_overlap, chunk_size // 2),
                count_tokens=self.dense.count_tokens,
                include_references=self.config.include_references,
            )
            timings["chunk"] = _ms(t)
            if not chunks:
                raise PermanentIngestionError(
                    "No extractable text found (scanned/image-only PDFs need OCR, not supported)"
                )

            with self.session_factory() as s:
                paper = s.get(Paper, paper_id)
                if paper is None:
                    return None
                payloads = [chunk_payload(paper, doc, c) for c in chunks]
            title = payloads[0]["title"]

            t = time.perf_counter()
            texts = [embedding_text(title, c.section, c.text) for c in chunks]
            dense_vectors = self.dense.embed_documents(texts)
            sparse_vectors = self.sparse.embed_documents(texts)
            timings["embed"] = _ms(t)

            t = time.perf_counter()
            self.store.ensure_collection(self.dense.dimension)
            self.store.delete_paper(paper_id)  # idempotency: drop vectors from earlier attempts
            vectors_written = True
            self.store.upsert(
                (
                    ChunkPoint(c.chunk_id, d, sp, pl)
                    for c, d, sp, pl in zip(
                        chunks, dense_vectors, sparse_vectors, payloads, strict=True
                    )
                ),
                batch_size=self.config.upsert_batch_size,
            )
            timings["index"] = _ms(t)

            if not self._finish(paper_id, job_id, doc, chunks, timings):
                self.store.delete_paper(paper_id)
                logger.info("Paper deleted during ingestion; vectors removed", extra=log_ctx)
                return None
        except Exception as exc:
            permanent = isinstance(exc, PermanentIngestionError)
            final = permanent or final_attempt
            logger.warning(
                "Ingestion failed",
                extra={**log_ctx, "error_type": type(exc).__name__, "final": final},
                exc_info=not permanent,
            )
            if vectors_written:
                self._compensate(paper_id)
            self._fail(paper_id, job_id, exc, final=final)
            if permanent:
                raise
            raise TransientIngestionError(str(exc)) from exc

        logger.info(
            "Ingestion succeeded",
            extra={**log_ctx, "chunks": len(chunks), **{f"ms_{k}": v for k, v in timings.items()}},
        )
        return IngestionResult(paper_id, doc.page_count, len(chunks), timings)

    def _compensate(self, paper_id: UUID) -> None:
        try:
            self.store.delete_paper(paper_id)
        except Exception:  # noqa: BLE001 - original error matters more; log and move on
            logger.error(
                "Could not remove partial vectors", extra={"paper_id": str(paper_id)}, exc_info=True
            )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
