"""Dense embeddings with Sentence Transformers (BGE by default).

The model is loaded lazily on first use and then reused for the life of the process, guarded by
a lock so concurrent requests don't load it twice. ``sentence_transformers`` (and torch) are
imported inside ``_load`` so that processes which never embed (e.g. migrations) stay light.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# BGE v1.5 recommends this prefix for short retrieval queries (not for documents).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseEmbedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def configure_model_cache(cache_dir: Path) -> None:
    """Point Hugging Face caches at ``cache_dir`` before any HF library is imported."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._batch_size = batch_size
        if query_instruction is None and "bge" in model_name.lower() and "-en" in model_name:
            query_instruction = BGE_QUERY_INSTRUCTION
        self._query_instruction = query_instruction or ""
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> Any:
        configure_model_cache(self._cache_dir)
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model", extra={"model": self.model_name})
        model = SentenceTransformer(
            self.model_name, device=self._device, cache_folder=str(self._cache_dir)
        )
        logger.info(
            "Embedding model loaded",
            extra={"model": self.model_name, "max_seq_length": model.max_seq_length},
        )
        return model

    @property
    def dimension(self) -> int:
        model = self.model
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            self._query_instruction + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()  # type: ignore[no-any-return]
