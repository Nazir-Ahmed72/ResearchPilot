"""BM25 sparse vectors for Qdrant.

Qdrant stores sparse vectors and, with the ``IDF`` modifier on the collection, computes inverse
document frequency from the indexed corpus at query time. So the document side only needs the
BM25 *term-frequency* component, and the query side is a binary bag of terms:

    doc weight(t)   = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
    query weight(t) = 1
    score(q, d)     = sum_t  idf(t) * doc_weight(t)        (computed by Qdrant)

Terms are lower-cased, stop-word filtered and Snowball-stemmed, then mapped to stable uint32
ids with CRC32 (no vocabulary file to persist or keep in sync).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

import snowballstemmer

_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "also",
        "et",
        "al",
        "via",
        "using",
        "use",
        "used",
    ]
)


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


class Bm25SparseEncoder:
    def __init__(self, *, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 256.0) -> None:
        self.k1 = k1
        self.b = b
        # A fixed average length keeps document vectors independent of the rest of the corpus,
        # so indexing one paper never requires re-encoding the others.
        self.avg_doc_len = avg_doc_len
        self._stemmer = snowballstemmer.stemmer("english")

    def tokenize(self, text: str) -> list[str]:
        tokens = [
            t
            for t in _TOKEN.findall(text.lower())
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())
        ]
        return self._stemmer.stemWords(tokens)  # type: ignore[no-any-return]

    @staticmethod
    def token_id(token: str) -> int:
        return zlib.crc32(token.encode("utf-8"))

    def _term_counts(self, text: str) -> tuple[Counter[int], int]:
        tokens = self.tokenize(text)
        return Counter(self.token_id(t) for t in tokens), len(tokens)

    def embed_document(self, text: str) -> SparseEmbedding:
        counts, length = self._term_counts(text)
        norm = self.k1 * (1 - self.b + self.b * length / self.avg_doc_len)
        items = sorted((i, tf * (self.k1 + 1) / (tf + norm)) for i, tf in counts.items())
        return SparseEmbedding([i for i, _ in items], [w for _, w in items])

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        return [self.embed_document(t) for t in texts]

    def embed_query(self, text: str) -> SparseEmbedding:
        counts, _ = self._term_counts(text)
        ids = sorted(counts)
        return SparseEmbedding(ids, [1.0] * len(ids))
