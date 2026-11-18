"""FastAPI dependency providers.

Route handlers receive infrastructure through these functions instead of importing globals, so
tests can swap any of them via ``app.dependency_overrides``. Authentication will plug in here as
another dependency without touching route logic.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.resources import Resources


def get_resources(request: Request) -> Resources:
    return request.app.state.resources  # type: ignore[no-any-return]


def get_settings_dep(request: Request) -> Settings:
    return get_resources(request).settings


def get_db(request: Request) -> Iterator[Session]:
    """One session per request; commit is explicit in services, rollback on any error."""
    session = get_resources(request).session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
