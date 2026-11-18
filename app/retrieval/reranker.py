"""Cross-encoder reranking.

Bi-encoders (BGE embeddings) encode query and passage separately, which is fast but coarse. A
cross-encoder reads query and passage *together* and scores relevance much more accurately, at a
cost per pair, so it only re-scores the top fused candidates.

Model loading and inference are isolated here; business logic only sees ``Reranker.score``.
Scores are the sigmoid of the model's logit, in 0..1. They are monotonic (good for ranking) but
NOT calibrated probabilities: measured with bge-reranker-base, a relevant passage that does not
name the query's subject can score ~0.001 while still ranking far above irrelevant text. Any
"is the evidence relevant?" threshold must be calibrated on real data, not assumed to be 0.5.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from app.embeddings.dense import configure_model_cache
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    configure_model_cache(self._cache_dir)
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker model", extra={"model": self.model_name})
                    self._model = CrossEncoder(
                        self.model_name,
                        device=self._device,
                        max_length=self._max_length,
                        cache_folder=str(self._cache_dir),
                    )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        import torch

        raw = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            activation_fn=torch.nn.Identity(),  # raw logits; sigmoid applied explicitly below
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [_sigmoid(float(x)) for x in raw]


def rerank_text(chunk: RetrievedChunk) -> str:
    """What the cross-encoder reads: same context header as used for embedding."""
    header = " | ".join(p for p in (chunk.title, chunk.section) if p)
    return f"{header}\n{chunk.text}" if header else chunk.text


def rerank(
    reranker: Reranker, query: str, chunks: Sequence[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    scores = reranker.score(query, [rerank_text(c) for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].chunk_id))
    return [
        chunks[i].with_stage("rerank", scores[i], rank)
        for rank, i in enumerate(order[:top_k], start=1)
    ]
