"""Ragas generation metrics (Ragas 0.4 ``metrics.collections`` API).

The judge LLM is reached through Groq's OpenAI-compatible endpoint, so no extra provider is
needed. Embeddings (for answer relevancy / correctness) reuse the project's BGE model.

Metrics and the inputs they need:
* faithfulness            question, answer, retrieved contexts
* answer_relevancy        question, answer                     (+ embeddings)
* answer_correctness      question, answer, reference answer   (+ embeddings)
* context_recall          question, retrieved contexts, reference answer
* context_precision       question, retrieved contexts, reference answer (one LLM call per
                          context: the most expensive metric, off by default)

Ragas makes several LLM calls per metric per question; on Groq's free tier evaluate a small
subset and pace the run. Failures of one metric are recorded as ``None``, never as 0.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.core.config import Settings
from app.embeddings.dense import DenseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "answer_correctness", "context_recall")
ALL_METRICS = (*DEFAULT_METRICS, "context_precision")
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"


def _embeddings_adapter(embedder: DenseEmbedder):  # type: ignore[no-untyped-def]
    from ragas.embeddings.base import BaseRagasEmbedding

    class BgeRagasEmbedding(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.embed_documents([text])[0]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return embedder.embed_documents(texts)

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts)

    return BgeRagasEmbedding()


def build_metrics(
    settings: Settings, embedder: DenseEmbedder, names: Sequence[str]
) -> dict[str, Any]:
    if settings.groq_api_key is None:
        raise RuntimeError("Ragas evaluation needs GROQ_API_KEY for the judge LLM")
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics import collections as rc

    client = AsyncOpenAI(
        api_key=settings.groq_api_key.get_secret_value(),
        base_url=GROQ_OPENAI_BASE_URL,
        max_retries=6,
    )
    llm = llm_factory(settings.llm_fast_model, provider="openai", client=client)
    embeddings = _embeddings_adapter(embedder)
    factories = {
        "faithfulness": lambda: rc.Faithfulness(llm=llm),
        "answer_relevancy": lambda: rc.AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": lambda: rc.AnswerCorrectness(llm=llm, embeddings=embeddings),
        "context_recall": lambda: rc.ContextRecall(llm=llm),
        "context_precision": lambda: rc.ContextPrecisionWithReference(llm=llm),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"Unknown Ragas metrics: {sorted(unknown)}")
    return {n: factories[n]() for n in names}


def _inputs(name: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    q, answer, contexts = row["question"], row.get("answer") or "", row.get("contexts") or []
    reference = row.get("reference_answer") or ""
    needs_reference = name in {"answer_correctness", "context_recall", "context_precision"}
    if needs_reference and not reference:
        return None  # e.g. unanswerable questions have no reference answer
    if name == "faithfulness":
        return (
            {"user_input": q, "response": answer, "retrieved_contexts": contexts}
            if contexts
            else None
        )
    if name == "answer_relevancy":
        return {"user_input": q, "response": answer}
    if name == "answer_correctness":
        return {"user_input": q, "response": answer, "reference": reference}
    if name == "context_recall":
        return {"user_input": q, "retrieved_contexts": contexts, "reference": reference}
    return {"user_input": q, "reference": reference, "retrieved_contexts": contexts}


async def _score_row(metrics: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for name, metric in metrics.items():
        inputs = _inputs(name, row)
        if inputs is None:
            continue
        try:
            result = await metric.ascore(**inputs)
            value = getattr(result, "value", result)
            scores[name] = round(float(value), 4) if value is not None else None
        except Exception as exc:  # noqa: BLE001 - a judge failure is "unknown", not a zero
            logger.warning(
                "Ragas metric failed",
                extra={
                    "metric": name,
                    "question": row.get("question_id"),
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                },
            )
            scores[name] = None
    return scores


def score_rows(
    rows: list[dict[str, Any]],
    settings: Settings,
    embedder: DenseEmbedder,
    names: Sequence[str] = DEFAULT_METRICS,
    pace_seconds: float = 0.0,
) -> None:
    """Adds ``row['ragas']`` to every completed row, sequentially, optionally pausing between
    rows: on Groq's free tier, unpaced runs exhaust the judge's rate limit and metrics fail."""
    metrics = build_metrics(settings, embedder, names)

    async def run() -> None:
        first = True
        for row in rows:
            if not row.get("completed"):
                continue
            if not first and pace_seconds:
                await asyncio.sleep(pace_seconds)
            first = False
            row["ragas"] = await _score_row(metrics, row)

    asyncio.run(run())
