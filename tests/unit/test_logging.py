from __future__ import annotations

import json
import logging

import pytest

from app.core.logging import REDACTED, ConsoleFormatter, JsonFormatter, redact, request_id_var


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, msg, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_request_id_and_extras() -> None:
    token = request_id_var.set("req-123")
    try:
        out = json.loads(JsonFormatter().format(_record("hello", paper_id="p1")))
    finally:
        request_id_var.reset(token)
    assert out["message"] == "hello"
    assert out["request_id"] == "req-123"
    assert out["paper_id"] == "p1"
    assert out["level"] == "INFO"


def test_json_formatter_omits_request_id_outside_requests() -> None:
    out = json.loads(JsonFormatter().format(_record("hello")))
    assert "request_id" not in out


def test_secret_like_keys_are_redacted() -> None:
    out = json.loads(JsonFormatter().format(_record("x", api_key="abc", groq_token="def")))
    assert out["api_key"] == REDACTED
    assert out["groq_token"] == REDACTED


def test_secret_like_values_are_redacted_in_messages() -> None:
    msg = "calling with gsk_ABCDEFGHIJKLMNOP and postgresql://user:hunter2@db:5432/x"
    out = json.loads(JsonFormatter().format(_record(msg)))
    assert "gsk_ABCDEFGHIJKLMNOP" not in out["message"]
    assert "hunter2" not in out["message"]


def test_redact_handles_nested_structures() -> None:
    data = {"outer": {"password": "p", "list": ["sk-abcdefghijklmnop", 3]}}
    assert redact(data) == {"outer": {"password": REDACTED, "list": [REDACTED, 3]}}


def test_console_formatter_is_readable() -> None:
    line = ConsoleFormatter().format(_record("hello", api_key="abc"))
    assert "hello" in line
    assert f"api_key={REDACTED}" in line


@pytest.mark.parametrize(
    "text",
    [
        "failed: api_key=supersecret",
        "Authorization failed, token: supersecret",
        '{"password": "supersecret"}',
        "connect?user=x&password=supersecret&ssl=1",
        "API-KEY = supersecret",
    ],
)
def test_key_value_secrets_are_redacted(text: str) -> None:
    out = redact(text)
    assert "supersecret" not in out
    assert REDACTED in out


def test_ordinary_text_is_untouched() -> None:
    text = "Tokenization of 5 passwords-free documents; secretary notes"
    assert redact(text) == text
