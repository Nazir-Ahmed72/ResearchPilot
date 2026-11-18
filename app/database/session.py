"""Engine and session management.

We use synchronous SQLAlchemy. Reasons: the Celery ingestion worker is synchronous, metadata
queries are small and fast, and FastAPI runs ``def`` endpoints in a thread pool, so the event
loop is never blocked. This avoids maintaining parallel sync/async data-access code.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, is_pooled_neon_url


def create_db_engine(settings: Settings) -> Engine:
    if settings.is_sqlite:
        db_path = make_url(settings.database_url).database
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            settings.database_url,
            echo=settings.database_echo,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fks(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            # SQLite ignores ON DELETE CASCADE unless foreign keys are switched on per connection.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        # Neon scale-to-zero closes idle connections: ping before use, recycle before the
        # compute's suspend timeout. The first query after a suspend pays a cold start.
        pool_pre_ping=True,
        pool_recycle=settings.database_pool_recycle_seconds,
        connect_args=postgres_connect_args(settings),
    )


def postgres_connect_args(settings: Settings) -> dict[str, object]:
    """psycopg connection options for Neon (or any PostgreSQL)."""
    args: dict[str, object] = {"connect_timeout": settings.database_connect_timeout_seconds}
    if is_pooled_neon_url(settings.database_url):
        # Neon's pooler is PgBouncer in transaction mode: a prepared statement created on one
        # server connection may not exist on the next. Disable psycopg's auto-preparation.
        args["prepare_threshold"] = None
    return args


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False lets API handlers read attributes after commit without new queries.
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope for workers/scripts: commit on success, roll back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
