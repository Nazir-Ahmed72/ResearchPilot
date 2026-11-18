"""Ingestion error hierarchy.

``PermanentIngestionError``: retrying cannot help (bad/empty PDF). The job fails immediately.
``TransientIngestionError``: an infrastructure hiccup (Qdrant/DB/network). Celery retries it.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for ingestion failures."""


class PermanentIngestionError(IngestionError):
    pass


class TransientIngestionError(IngestionError):
    pass


class PdfValidationError(PermanentIngestionError):
    """Raised at upload time. ``code`` maps to an HTTP status in the API layer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
