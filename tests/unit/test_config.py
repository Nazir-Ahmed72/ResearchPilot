from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings, is_pooled_neon_url, normalise_database_url


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults_use_groq_and_local_services() -> None:
    s = make()
    assert s.llm_provider == "groq"
    assert s.api_prefix == "/api/v1"
    assert s.langsmith_tracing is False
    assert s.max_upload_size_bytes == 50 * 1024 * 1024


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "5")
    s = make()
    assert s.llm_model == "some-other-model"
    assert s.max_upload_size_bytes == 5 * 1024 * 1024


def test_secrets_are_not_exposed_in_repr() -> None:
    s = make(groq_api_key=SecretStr("gsk_supersecretvalue123"))
    assert "gsk_supersecretvalue123" not in repr(s)
    assert "gsk_supersecretvalue123" not in str(s.model_dump())
    assert s.groq_api_key is not None
    assert s.groq_api_key.get_secret_value() == "gsk_supersecretvalue123"


@pytest.mark.parametrize(("raw", "expected"), [("api/v1/", "/api/v1"), ("/", ""), ("/x", "/x")])
def test_api_prefix_is_normalised(raw: str, expected: str) -> None:
    assert make(api_prefix=raw).api_prefix == expected


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make(chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_tracing_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        make(langsmith_tracing=True)


def test_unknown_llm_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        make(llm_provider="not-a-provider")


NEON_POOLED = (
    "postgresql://u:p@ep-cool-name-a1b2c3-pooler.us-east-2.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)
NEON_DIRECT = NEON_POOLED.replace("-pooler", "")


def test_default_database_is_local_sqlite() -> None:
    assert make().is_sqlite
    assert not make(database_url=NEON_POOLED).is_sqlite


@pytest.mark.parametrize("scheme", ["postgres://", "postgresql://", "postgresql+psycopg://"])
def test_neon_urls_are_normalised_to_psycopg(scheme: str) -> None:
    url = normalise_database_url(f"{scheme}u:p@host/db?sslmode=require")
    assert url == "postgresql+psycopg://u:p@host/db?sslmode=require"


def test_query_string_survives_normalisation() -> None:
    s = make(database_url=NEON_POOLED)
    assert s.database_url.startswith("postgresql+psycopg://")
    assert s.database_url.endswith("?sslmode=require&channel_binding=require")


def test_migrations_prefer_direct_endpoint() -> None:
    s = make(database_url=NEON_POOLED, database_url_unpooled=NEON_DIRECT)
    assert "-pooler" not in s.migration_database_url
    assert make(database_url=NEON_POOLED).migration_database_url == s.database_url


@pytest.mark.parametrize(
    ("url", "pooled"),
    [
        (NEON_POOLED, True),
        (NEON_DIRECT, False),
        ("postgresql+psycopg://u:p@localhost:5432/db", False),
        ("postgresql+psycopg://u:p-pooler@host/db", False),  # "-pooler" in password, not host
    ],
)
def test_pooled_endpoint_detection(url: str, pooled: bool) -> None:
    assert is_pooled_neon_url(url) is pooled


def test_empty_secret_values_mean_not_set() -> None:
    s = make(qdrant_api_key="", groq_api_key="  ", langsmith_api_key="")
    assert s.qdrant_api_key is None and s.groq_api_key is None and s.langsmith_api_key is None
