"""Application configuration.

All settings come from environment variables (or a local ``.env`` file) so the same code runs
locally, in tests, in Docker and in the cloud. Secrets use ``SecretStr`` so they never appear in
``repr()``, logs or tracebacks by accident.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Environment = Literal["local", "test", "staging", "production"]
LLMProvider = Literal["groq", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -------------------------------------------------------------------------
    app_name: str = "ResearchPilot"
    app_version: str = "0.1.0"
    environment: Environment = "local"
    debug: bool = False
    api_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True

    # --- Database (Neon serverless PostgreSQL) ------------------------------------------------
    # DATABASE_URL: Neon *pooled* endpoint (host contains "-pooler"), used by the application.
    # DATABASE_URL_UNPOOLED: Neon *direct* endpoint, used by Alembic migrations, which need
    # session state that PgBouncer transaction pooling does not provide. Falls back to
    # DATABASE_URL when unset. Plain "postgresql://" URLs from the Neon console are accepted.
    # The SQLite default lets the API start with no database configured (development/tests).
    database_url: str = "sqlite:///./data/researchpilot.db"
    database_url_unpooled: str | None = None
    database_echo: bool = False
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=5, ge=0)
    # Neon suspends idle computes (default 5 min) and closes their connections; recycle sooner.
    database_pool_recycle_seconds: int = Field(default=240, gt=0)
    database_connect_timeout_seconds: int = Field(default=10, gt=0)

    # --- Redis -------------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- Qdrant ------------------------------------------------------------------------------
    # An http(s) URL for a Qdrant server, ":memory:" for an in-process instance, or a filesystem
    # path for Qdrant's embedded local mode (useful for development without Docker).
    qdrant_location: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "researchpilot_chunks"

    # --- LLM ---------------------------------------------------------------------------------
    llm_provider: LLMProvider = "groq"
    # Main model for reasoning-heavy steps (synthesis, verification).
    llm_model: str = "openai/gpt-oss-120b"
    # Smaller, faster model for frequent lightweight steps (query transformation).
    llm_fast_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    groq_api_key: SecretStr | None = None

    # --- Embeddings / reranking --------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_max_length: int = Field(default=512, gt=0)
    # Model downloads go here (not the user profile). Relative paths resolve to the project root.
    hf_home: Path = Path(".cache/hf")

    # --- Ingestion ---------------------------------------------------------------------------
    upload_dir: Path = Path("data/uploads")
    # Where Python/Starlette spool temporary files (large multipart uploads). None = OS default.
    temp_dir: Path | None = None
    max_upload_size_mb: int = Field(default=50, gt=0)
    max_files_per_upload: int = Field(default=10, gt=0)
    max_pdf_pages: int = Field(default=500, gt=0)
    # BGE-small reads at most 512 tokens; chunks are embedded with a title/section header, so
    # the default leaves headroom. The pipeline also caps chunks to the model's real limit.
    chunk_size_tokens: int = Field(default=350, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    index_references: bool = False
    qdrant_upsert_batch_size: int = Field(default=64, gt=0)

    # --- Retrieval ---------------------------------------------------------------------------
    retrieval_top_k: int = Field(default=8, gt=0, le=100)
    # Candidates fetched from each retriever (dense, sparse) per query before fusion.
    retrieval_candidate_k: int = Field(default=30, gt=0, le=200)
    # How many fused candidates the cross-encoder re-scores (the expensive step on CPU).
    # Measured on an 8-core laptop CPU, bge-reranker-base: 10 pairs ~7.5 s, 20 pairs ~12.8 s.
    rerank_candidate_k: int = Field(default=10, gt=0, le=200)
    rrf_k: int = Field(default=60, gt=0)
    max_query_variants: int = Field(default=4, ge=1, le=8)

    # --- Background tasks --------------------------------------------------------------------
    # "celery": separate worker process via Redis. "inline": run in the API process (dev/tests).
    task_backend: Literal["celery", "inline"] = "celery"
    # Seconds an idle worker blocks on Redis before re-polling. Larger = fewer Redis commands
    # (matters on Upstash's free tier); new tasks are still delivered immediately.
    celery_broker_polling_interval: float = Field(default=10.0, gt=0)
    ingestion_max_retries: int = Field(default=3, ge=0)

    # --- Observability -----------------------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "researchpilot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Health checks -----------------------------------------------------------------------
    # Neon cold starts (scale-to-zero resume) take ~2-3 s; keep the timeout above that.
    health_check_timeout_seconds: float = Field(default=5.0, gt=0)

    @field_validator("database_url", "database_url_unpooled")
    @classmethod
    def _use_psycopg_driver(cls, value: str | None) -> str | None:
        return normalise_database_url(value) if value else value

    @field_validator("upload_dir", "hf_home", "temp_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path | None) -> Path | None:
        if value is None or value.is_absolute():
            return value
        return PROJECT_ROOT / value

    @field_validator("api_prefix")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        value = "/" + value.strip("/")
        return "" if value == "/" else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError("LANGSMITH_TRACING=true requires LANGSMITH_API_KEY")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def migration_database_url(self) -> str:
        return self.database_url_unpooled or self.database_url


def normalise_database_url(url: str) -> str:
    """Make Neon/Heroku-style URLs use SQLAlchemy's psycopg (v3) driver."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def is_pooled_neon_url(url: str) -> bool:
    """True for Neon's PgBouncer endpoint (hostname label ends in ``-pooler``)."""
    from sqlalchemy.engine import make_url  # local import keeps config import-light

    host = make_url(url).host or ""
    return host.split(".", 1)[0].endswith("-pooler")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached; call ``cache_clear()`` in tests)."""
    return Settings()
