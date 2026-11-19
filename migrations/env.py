"""Alembic environment.

The database URL comes from application settings, never from alembic.ini, so credentials are not
committed. Migrations use Neon's *direct* endpoint (``DATABASE_URL_UNPOOLED``) because the pooled
endpoint (PgBouncer, transaction mode) breaks session-level DDL behaviour. A caller (e.g.
tests) may override the URL via ``config.attributes["database_url"]``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import get_settings, normalise_database_url
from app.database.models import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

# Tables owned by LangGraph's PostgresSaver (created by its own setup()); Alembic must neither
# create nor drop them.
EXTERNAL_TABLES = {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}


def include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    if type_ == "table" and name in EXTERNAL_TABLES:
        return False
    table = getattr(obj, "table", None)
    return not (table is not None and table.name in EXTERNAL_TABLES)


def _database_url() -> str:
    url = config.attributes.get("database_url") or get_settings().migration_database_url
    return normalise_database_url(url)


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
        render_as_batch=url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _database_url()
    connectable = create_engine(url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
            # SQLite cannot ALTER most constraints; batch mode recreates tables instead.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
