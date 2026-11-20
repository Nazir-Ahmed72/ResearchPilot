"""Streamlit dashboard: every page renders, and failures are shown as readable messages.

Skipped when Streamlit is not installed (it is an optional `ui` extra, not needed by CI).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "ui" / "app.py")
PAGES = ["🏠 Overview", "📄 Papers", "🔎 Search", "🧠 Research", "📊 Evaluation", "❓ Help"]


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("RESEARCHPILOT_API_URL", "http://127.0.0.1:1")  # nothing listens here
    return AppTest.from_file(APP, default_timeout=30)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_when_api_is_down(app: AppTest, page: str) -> None:
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
    assert any("API not running" in e.value for e in app.sidebar.error)


def test_overview_explains_how_to_start_the_api(app: AppTest) -> None:
    app.run()
    assert not app.exception
    assert any("Cannot reach the API" in e.value for e in app.error)


def _client(monkeypatch: pytest.MonkeyPatch, handler) -> None:  # type: ignore[no-untyped-def]
    real = httpx.Client

    def fake(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake)


def test_api_errors_carry_the_server_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("dashboard_module", APP)
    assert spec and spec.loader
    source = Path(APP).read_text(encoding="utf-8").replace("\nmain()\n", "\n")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, APP, "exec"), module.__dict__)  # noqa: S102 - load without running UI

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(503, json={"status": "degraded", "checks": {}})
        if request.url.path.endswith("/research/query"):
            return httpx.Response(
                503,
                json={
                    "error": {
                        "code": "dependency_unavailable",
                        "message": "Task queue is unavailable",
                    }
                },
            )
        return httpx.Response(
            404, json={"error": {"message": "Paper not found", "details": {"paper_id": "x"}}}
        )

    _client(monkeypatch, handler)
    api = module.Api("http://api")
    assert api.get("/health")["status"] == "degraded"  # 503 health report is data, not an error
    with pytest.raises(module.ApiError, match="Task queue is unavailable"):
        api.post("/research/query", json={})  # any other 503 is an error
    with pytest.raises(module.ApiError, match="Paper not found") as err:
        api.get("/papers/x")
    assert err.value.args[1] == {"paper_id": "x"}
