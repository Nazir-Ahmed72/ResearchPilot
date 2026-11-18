"""Alembic migrations: upgrade/downgrade work and the schema matches the ORM models.

Runs on SQLite always, and additionally on PostgreSQL when TEST_DATABASE_URL is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from app.core.config import normalise_database_url
from app.database.models import Base

ROOT = Path(__file__).resolve().parents[2]


def _config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = url
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture(params=["sqlite", "postgres"])
def database_url(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    if request.param == "sqlite":
        return f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set; PostgreSQL migration test skipped")
    return normalise_database_url(url)


def test_upgrade_matches_models_and_downgrade_cleans_up(database_url: str) -> None:
    cfg = _config(database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            diff = compare_metadata(
                MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata
            )
        assert diff == [], f"Models and migrations are out of sync: {diff}"
        assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())

        command.downgrade(cfg, "base")
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
        assert remaining == set()
    finally:
        engine.dispose()
