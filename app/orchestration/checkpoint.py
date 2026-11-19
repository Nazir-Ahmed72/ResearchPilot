"""LangGraph checkpointer construction.

Production uses ``PostgresSaver`` on Neon's DIRECT endpoint: checkpoint writes use prepared
statements and session state that PgBouncer transaction pooling (the ``-pooler`` URL) does not
support. ``setup()`` creates LangGraph's own checkpoint tables idempotently; they are managed by
LangGraph, not by our Alembic migrations.

Connections are pooled and health-checked before use, because Neon closes idle connections when
the compute scales to zero.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings

logger = logging.getLogger(__name__)


def to_libpq_url(sqlalchemy_url: str) -> str:
    """'postgresql+psycopg://...' -> 'postgresql://...' (psycopg/libpq connection string)."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def create_checkpointer(settings: Settings) -> tuple[BaseCheckpointSaver, Callable[[], None]]:
    """Return (checkpointer, close)."""
    if settings.research_checkpointer == "memory" or settings.is_sqlite:
        return InMemorySaver(), lambda: None

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=to_libpq_url(settings.migration_database_url),
        min_size=1,
        max_size=4,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=ConnectionPool.check_connection,
        open=True,
    )
    saver = PostgresSaver(pool)  # type: ignore[arg-type]
    saver.setup()
    logger.info("PostgreSQL checkpointer ready")
    return saver, pool.close
