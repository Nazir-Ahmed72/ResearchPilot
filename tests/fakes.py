"""Test doubles. Deterministic and dependency-free; never used by application code."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from uuid import UUID


class FakeDenseEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    model_name = "fake-embedder"

    def __init__(self, dimension: int = 32, max_tokens: int = 512) -> None:
        self._dimension = dimension
        self._max_tokens = max_tokens
        self.document_calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self._dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class RecordingDispatcher:
    """Records enqueued jobs instead of running them; tests run the pipeline explicitly."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[UUID, UUID]] = []
        self.fail = fail

    def enqueue_ingestion(self, paper_id: UUID, job_id: UUID) -> str:
        from app.core.exceptions import DependencyUnavailableError

        if self.fail:
            raise DependencyUnavailableError("Task queue (Redis) is unavailable")
        self.calls.append((paper_id, job_id))
        return f"task-{len(self.calls)}"

    def close(self) -> None:
        pass


class KeywordReranker:
    """Scores by query-word overlap; deterministic stand-in for a cross-encoder."""

    model_name = "fake-reranker"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        q = {w.strip("?.,").lower() for w in query.split() if len(w) > 3}
        out = []
        for p in passages:
            words = {w.strip("?.,").lower() for w in p.split()}
            out.append(len(q & words) / (len(q) or 1))
        return out


class FakePlanner:
    """Returns a fixed QueryPlan (or raises), recording the inputs it received."""

    def __init__(self, plan=None, error: Exception | None = None) -> None:  # type: ignore[no-untyped-def]
        self.plan = plan
        self.error = error
        self.inputs: list[dict] = []

    def invoke(self, input: dict, config=None):  # type: ignore[no-untyped-def]  # noqa: A002
        self.inputs.append(input)
        if self.error:
            raise self.error
        return self.plan
