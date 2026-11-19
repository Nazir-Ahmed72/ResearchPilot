"""Container health check for the Celery worker: ``python -m app.workers.healthcheck``.

Celery's usual ``celery inspect ping`` needs remote control, which is disabled to minimise Redis
traffic. Instead the worker writes a ready file when it has started (and removes it on shutdown),
and this check confirms both that file and that the broker is reachable.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

READY_FILE = Path(tempfile.gettempdir()) / "researchpilot-worker.ready"


def main() -> int:
    if not READY_FILE.exists():
        print("worker not ready")
        return 1
    from redis import Redis

    from app.core.config import get_settings

    try:
        Redis.from_url(get_settings().redis_url, socket_connect_timeout=3, socket_timeout=3).ping()
    except Exception as exc:  # noqa: BLE001
        print(f"broker unreachable: {type(exc).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
