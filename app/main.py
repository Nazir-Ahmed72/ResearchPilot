"""FastAPI application factory.

Run locally with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app.api.routes import health, papers, research, retrieval
from app.core.config import Settings, get_settings
from app.core.exceptions import ErrorResponse, register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.resources import Resources

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, resources: Resources | None = None) -> FastAPI:
    """Build the application.

    ``resources`` may be injected (tests); otherwise they are created from ``settings`` at
    startup and closed at shutdown.
    """
    settings = settings or (resources.settings if resources else get_settings())
    configure_logging(settings.log_level, settings.log_json)
    if settings.temp_dir is not None:
        # Large multipart uploads spool to disk; keep them on the configured drive.
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(settings.temp_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = resources is None
        app.state.resources = resources or Resources.from_settings(settings)
        logger.info(
            "Application started",
            extra={"version": settings.app_version, "environment": settings.environment},
        )
        try:
            yield
        finally:
            if owned:
                app.state.resources.close()
            logger.info("Application stopped")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Intelligent research paper analysis: hybrid retrieval, reranking and "
            "LangGraph-orchestrated, citation-verified research synthesis."
        ),
        lifespan=lifespan,
        # Documented default error shape for every route.
        responses={
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal error"},
        },
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    api = APIRouter(prefix=settings.api_prefix)
    api.include_router(health.router)
    api.include_router(papers.router)
    api.include_router(retrieval.router)
    api.include_router(research.router)
    app.include_router(api)
    return app


def __getattr__(name: str) -> FastAPI:
    # Lazily build the ASGI app for ``uvicorn app.main:app`` so importing this module (e.g. in
    # tests or Alembic) does not read settings or configure logging as a side effect.
    if name == "app":
        return create_app()
    raise AttributeError(name)
