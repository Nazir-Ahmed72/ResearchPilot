"""Provider-independent chat model construction.

Everything else depends on LangChain's ``BaseChatModel`` interface, so supporting another
provider means adding one branch here. Model names come from settings, never from code.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings
from app.core.exceptions import DependencyUnavailableError

ModelTier = Literal["main", "fast"]


class LLMConfigurationError(DependencyUnavailableError):
    code = "llm_not_configured"


def create_chat_model(settings: Settings, tier: ModelTier = "main") -> BaseChatModel:
    model = settings.llm_model if tier == "main" else settings.llm_fast_model
    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise LLMConfigurationError("GROQ_API_KEY is not set")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    raise LLMConfigurationError(
        f"LLM provider {settings.llm_provider!r} has no real model; tests inject fakes instead"
    )
