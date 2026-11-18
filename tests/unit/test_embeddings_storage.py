from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.embeddings.sparse import Bm25SparseEncoder
from app.ingestion.storage import LocalFileStorage

# --- BM25 sparse encoder ----------------------------------------------------------------------


def test_tokenizer_lowercases_removes_stopwords_and_stems() -> None:
    enc = Bm25SparseEncoder()
    assert enc.tokenize("The Retrieval of retrieved documents is retrieving!") == [
        "retriev",
        "retriev",
        "document",
        "retriev",
    ]


def test_document_vector_is_deterministic_sorted_and_positive() -> None:
    enc = Bm25SparseEncoder()
    a = enc.embed_document("dense retrieval and sparse retrieval")
    b = enc.embed_document("dense retrieval and sparse retrieval")
    assert a == b
    assert a.indices == sorted(a.indices)
    assert len(a.indices) == len(set(a.indices)) == 3  # dense, retriev, spars
    assert all(v > 0 for v in a.values)


def test_repeated_terms_saturate() -> None:
    enc = Bm25SparseEncoder()
    tid = enc.token_id("rag")
    once = dict(zip(*vars(enc.embed_document("rag model")).values(), strict=True))[tid]
    many = dict(zip(*vars(enc.embed_document("rag rag rag rag model")).values(), strict=True))[tid]
    assert once < many < once * (enc.k1 + 1)  # BM25 tf saturation bound


def test_longer_documents_get_lower_term_weight() -> None:
    enc = Bm25SparseEncoder()
    tid = enc.token_id("graph")
    short = enc.embed_document("graph")
    long = enc.embed_document("graph " + "filler " * 300)
    w_short = short.values[short.indices.index(tid)]
    w_long = long.values[long.indices.index(tid)]
    assert w_long < w_short


def test_query_vector_is_binary_bag_of_terms() -> None:
    q = Bm25SparseEncoder().embed_query("graph graph retrieval")
    assert q.values == [1.0, 1.0]


def test_empty_text_gives_empty_vector() -> None:
    assert Bm25SparseEncoder().embed_document("the of and").indices == []


# --- file storage -----------------------------------------------------------------------------


def test_storage_roundtrip_and_delete(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "files")
    pid = uuid.uuid4()
    key = storage.save(pid, b"%PDF-data")
    assert key == f"{pid}.pdf"
    assert storage.read(key) == b"%PDF-data"
    storage.delete(key)
    assert not storage.exists(key)
    storage.delete(key)  # idempotent


@pytest.mark.parametrize("key", ["../secret.pdf", "..\\x.pdf", "a/b.pdf", "evil.exe", ""])
def test_storage_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError):
        LocalFileStorage(tmp_path).read(key)
