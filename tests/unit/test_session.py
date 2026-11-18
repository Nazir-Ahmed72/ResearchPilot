"""Engine configuration for Neon endpoints (no network: engines connect lazily)."""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.database.session import create_db_engine, postgres_connect_args

POOLED = "postgresql://u:p@ep-x-123-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require"
DIRECT = POOLED.replace("-pooler", "")


def _settings(url: str) -> Settings:
    return Settings(_env_file=None, database_url=url)  # type: ignore[call-arg]


def test_pooled_endpoint_disables_prepared_statements() -> None:
    args = postgres_connect_args(_settings(POOLED))
    assert args["prepare_threshold"] is None
    assert args["connect_timeout"] == 10


def test_direct_endpoint_keeps_prepared_statements() -> None:
    assert "prepare_threshold" not in postgres_connect_args(_settings(DIRECT))


def test_postgres_engine_handles_scale_to_zero() -> None:
    engine = create_db_engine(_settings(POOLED))
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.url.query["sslmode"] == "require"
        assert engine.pool._pre_ping is True
        assert engine.pool._recycle == 240
    finally:
        engine.dispose()


def test_sqlite_parent_directory_is_created(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dir" / "app.db"
    engine = create_db_engine(_settings(f"sqlite:///{db.as_posix()}"))
    try:
        with engine.connect():
            pass
        assert db.exists()
    finally:
        engine.dispose()
