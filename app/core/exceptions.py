"""Domain exceptions and centralised FastAPI exception handlers.

Every error leaves the API in the same shape (``ErrorResponse``), so clients only need one
parser and every error carries the request ID needed to find it in the logs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_var

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class AppError(Exception):
    """Base class for expected, client-visible application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class InvalidInputError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "invalid_input"


class PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"


class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class DependencyUnavailableError(AppError):
    """An infrastructure dependency (DB, Qdrant, Redis, LLM provider) could not be reached."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "dependency_unavailable"


def error_response(
    status_code: int, code: str, message: str, details: Any | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=jsonable_encoder(details) if details is not None else None,
            request_id=request_id_var.get(),
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    log = logger.error if exc.status_code >= 500 else logger.info
    log("Application error", extra={"error_code": exc.code, "status_code": exc.status_code})
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def _validation_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "validation_error",
        "Request validation failed",
        errors,
    )


async def _http_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail))


def internal_error_response() -> JSONResponse:
    """Generic 500 body: details go to the logs only, never to clients."""
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "An unexpected error occurred"
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for expected errors.

    Unexpected exceptions are handled by ``RequestContextMiddleware`` rather than an
    ``Exception`` handler, because Starlette runs those outside user middleware, after the
    request context (request ID) has been torn down.
    """
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
