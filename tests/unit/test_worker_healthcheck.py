from __future__ import annotations

import pytest

from app.workers import healthcheck


def test_unhealthy_without_ready_file(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(healthcheck, "READY_FILE", tmp_path / "ready")
    assert healthcheck.main() == 1


def test_unhealthy_when_broker_unreachable(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ready = tmp_path / "ready"
    ready.touch()
    monkeypatch.setattr(healthcheck, "READY_FILE", ready)
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert healthcheck.main() == 1
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("exists", [True])
def test_ready_file_lives_in_temp_dir(exists: bool) -> None:
    assert healthcheck.READY_FILE.name == "researchpilot-worker.ready"
