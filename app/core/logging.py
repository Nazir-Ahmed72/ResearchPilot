"""Structured logging.

Uses the standard library ``logging`` module with a small JSON formatter instead of an extra
dependency. A ``contextvars`` variable carries the current request ID so every log line written
while handling a request (including inside worker threads started via ``asyncio.to_thread``) can
be correlated.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes present on every LogRecord; anything else was passed via ``extra=`` and is emitted.
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}

# Whole key names (or their suffix after _ / -), so "access_token" and "api_key" are masked but
# counters such as "input_tokens" / "max_tokens" are not.
_SECRET_KEY_PATTERN = re.compile(
    r"(^|[_-])(api[_-]?key|access[_-]?key|token|secret|password|passwd|authorization|cookie)$",
    re.I,
)
REDACTED = "***REDACTED***"
# (pattern, replacement) pairs applied to every logged/stored string.
_SECRET_VALUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # key=value / key: value / "key": "value" where the key name looks secret; keep the key.
    (
        re.compile(
            r"(?i)(\b(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|pwd)"
            r"[\"']?\s*[=:]\s*[\"']?)[^\s\"',;&]+"
        ),
        r"\g<1>" + REDACTED,
    ),
    (re.compile(r"gsk_[A-Za-z0-9]{10,}"), REDACTED),  # Groq keys
    (re.compile(r"lsv2_[A-Za-z0-9_]{10,}"), REDACTED),  # LangSmith keys
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), REDACTED),  # OpenAI/Anthropic-style keys
    (re.compile(r"(?<=://)[^:/@\s]+:[^@/\s]+(?=@)"), REDACTED),  # user:password in URLs
]


def redact(value: Any, key: str | None = None) -> Any:
    """Recursively mask values that look like credentials."""
    if key is not None and _SECRET_KEY_PATTERN.search(key):
        return REDACTED
    if isinstance(value, str):
        for pattern, replacement in _SECRET_VALUE_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = redact(value, key)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get()
        prefix = f"[{request_id[:8]}] " if request_id else ""
        extras = {
            k: redact(v, k)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        line = f"{self.formatTime(record)} {record.levelname:<8} {record.name}: {prefix}"
        line += redact(record.getMessage())
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + redact(self.formatException(record.exc_info))
        return line


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger once; safe to call repeatedly (e.g. in tests)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Uvicorn's access log duplicates our request logging middleware.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
