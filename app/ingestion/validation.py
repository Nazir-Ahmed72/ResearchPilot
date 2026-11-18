"""Upload validation: cheap checks first, then a real parse with PyMuPDF.

The declared MIME type is client-controlled and only used to reject obvious mismatches; the
authoritative checks are the ``%PDF-`` magic bytes and PyMuPDF actually opening the file.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

import pymupdf

from app.ingestion.errors import PdfValidationError

ALLOWED_CONTENT_TYPES = frozenset(
    {"application/pdf", "application/x-pdf", "application/octet-stream", ""}
)
PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class ValidatedPdf:
    filename: str
    page_count: int
    size_bytes: int


def sanitize_filename(raw: str | None) -> str:
    """Keep only a safe display name: no directories, control characters or odd lengths.

    The result is stored as metadata only; files on disk are named by generated UUIDs, so a
    hostile name can never influence a filesystem path.
    """
    name = raw or "upload.pdf"
    name = PureWindowsPath(PurePosixPath(name).name).name  # strip both / and \\ components
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name or "upload.pdf")[:255]


def validate_pdf(
    data: bytes,
    *,
    filename: str | None,
    content_type: str | None,
    max_size_bytes: int,
    max_pages: int,
) -> ValidatedPdf:
    safe_name = sanitize_filename(filename)
    if not safe_name.lower().endswith(".pdf"):
        raise PdfValidationError("unsupported_media_type", "Only .pdf files are accepted")
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type not in ALLOWED_CONTENT_TYPES:
        raise PdfValidationError(
            "unsupported_media_type", f"Content type {media_type!r} is not a PDF"
        )
    if len(data) == 0:
        raise PdfValidationError("invalid_pdf", "File is empty")
    if len(data) > max_size_bytes:
        raise PdfValidationError(
            "payload_too_large", f"File exceeds the {max_size_bytes // (1024 * 1024)} MB limit"
        )
    if not data.lstrip()[:1024].startswith(PDF_MAGIC) and PDF_MAGIC not in data[:1024]:
        raise PdfValidationError("invalid_pdf", "File does not have a PDF signature")

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass or doc.is_encrypted:
                raise PdfValidationError("invalid_pdf", "Encrypted PDFs are not supported")
            page_count = doc.page_count
    except PdfValidationError:
        raise
    except Exception as exc:  # PyMuPDF raises several types for corrupt input
        raise PdfValidationError("invalid_pdf", "File could not be parsed as a PDF") from exc

    if page_count == 0:
        raise PdfValidationError("invalid_pdf", "PDF has no pages")
    if page_count > max_pages:
        raise PdfValidationError(
            "invalid_pdf", f"PDF has {page_count} pages; the limit is {max_pages}"
        )
    return ValidatedPdf(filename=safe_name, page_count=page_count, size_bytes=len(data))
