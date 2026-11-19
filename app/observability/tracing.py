"""LangSmith tracing.

Tracing is opt-in (``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY``) and explicit: instead of
switching on LangChain's global env-var tracing, a ``LangChainTracer`` is attached to the
callbacks of the runs we choose (research workflow, evaluations). Tests and local development
therefore never need a LangSmith account, and nothing is sent unless configured.

Privacy: every run's inputs and outputs pass through the same ``redact`` used for logs, so API
keys, passwords and connection-string credentials never leave the process. Paper text is sent
when tracing is enabled (that is the point of tracing RAG); disable tracing for confidential
collections.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import Settings
from app.core.logging import redact

logger = logging.getLogger(__name__)


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = redact(payload)
    return redacted if isinstance(redacted, dict) else {"value": redacted}


def create_tracer(settings: Settings) -> BaseCallbackHandler | None:
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        return None
    from langchain_core.tracers.langchain import LangChainTracer
    from langsmith import Client

    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
        hide_inputs=redact_payload,
        hide_outputs=redact_payload,
    )
    logger.info("LangSmith tracing enabled", extra={"project": settings.langsmith_project})
    return LangChainTracer(project_name=settings.langsmith_project, client=client)


def langsmith_client(settings: Settings):  # type: ignore[no-untyped-def]
    """A LangSmith client for datasets/experiments (requires an API key, tracing flag optional)."""
    if settings.langsmith_api_key is None:
        raise RuntimeError("LANGSMITH_API_KEY is not set")
    from langsmith import Client

    return Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
        hide_inputs=redact_payload,
        hide_outputs=redact_payload,
    )


def flush(tracer: BaseCallbackHandler | None) -> None:
    """Block until queued traces are uploaded (call at the end of a run / on shutdown)."""
    if tracer is None:
        return
    try:
        tracer.wait_for_futures()  # type: ignore[attr-defined]
        client = getattr(tracer, "client", None)
        if client is not None:
            client.flush()
    except Exception:  # noqa: BLE001 - tracing must never break the application
        logger.warning("Could not flush LangSmith traces", exc_info=True)
