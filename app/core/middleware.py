"""ASGI middleware.

Implemented as plain ASGI middleware (not ``BaseHTTPMiddleware``) so it does not buffer
streaming responses, which the research endpoints will use later.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.exceptions import internal_error_response
from app.core.logging import request_id_var

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestContextMiddleware:
    """Assigns a request ID, exposes it in the response, and logs one line per request.

    A client-supplied ``X-Request-ID`` is reused only if it is short and contains safe
    characters; otherwise a new UUID is generated (prevents log injection).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for name, value in scope.get("headers", []):
            if name.decode("latin-1").lower() == REQUEST_ID_HEADER.lower():
                incoming = value.decode("latin-1")
                break
        request_id = incoming if incoming and _VALID_REQUEST_ID.match(incoming) else None
        request_id = request_id or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id

        status_code = 500
        response_started = False
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("Unhandled exception")
            if response_started:
                raise
            await internal_error_response()(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request completed",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            request_id_var.reset(token)
