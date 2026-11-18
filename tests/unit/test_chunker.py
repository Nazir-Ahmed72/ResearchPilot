from __future__ import annotations

import pytest

from app.ingestion.chunker import (
    FRONT_MATTER,
    build_sections,
    chunk_document,
    chunk_id_for,
    word_count_tokens,
)
from app.ingestion.pdf_loader import TextBlock, load_pdf
from app.ingestion.text_cleaner import clean_blocks
from tests.pdf_factory import make_paper_pdf


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}." if i % 12 == 11 else f"{prefix}{i}" for i in range(n))


def count(text: str) -> int:  # 1 token per word: makes budgets easy to reason about
    return len(text.split())


def para(page: int, text: str, heading: bool = False) -> TextBlock:
    return TextBlock(page, text, 10, heading, 100, 110, 792, is_heading=heading)


def test_sections_split_at_headings_with_front_matter() -> None:
    sections = build_sections(
        [
            para(1, "Title stuff"),
            para(1, "Intro", True),
            para(1, "a"),
            para(2, "Method", True),
            para(2, "b"),
        ]
    )
    assert [s.title for s in sections] == [FRONT_MATTER, "Intro", "Method"]
    assert sections[2].paragraphs == [(2, "b")]


def test_chunks_respect_budget_and_never_cross_sections() -> None:
    blocks = [para(1, "Intro", True)] + [para(1, words(40, "a")) for _ in range(5)]
    blocks += [para(2, "Method", True), para(2, words(30, "b"))]
    chunks = chunk_document(
        blocks, document_key="k", chunk_size=100, overlap=20, count_tokens=count
    )
    assert all(c.token_count <= 100 for c in chunks)
    assert {c.section for c in chunks} == {"Intro", "Method"}
    assert not any("a1" in c.text and "b1" in c.text for c in chunks)
    method = [c for c in chunks if c.section == "Method"]
    assert len(method) == 1 and method[0].page_start == method[0].page_end == 2


def test_overlap_repeats_trailing_text_between_neighbours() -> None:
    sentences = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_document(
        [para(1, "S", True), para(1, sentences)],
        document_key="k",
        chunk_size=50,
        overlap=12,
        count_tokens=count,
    )
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail = prev.text.split(". ")[-1]
        assert tail.rstrip(".") in nxt.text


def test_zero_overlap_has_no_repetition() -> None:
    sentences = " ".join(f"Sentence {i} ends." for i in range(40))
    chunks = chunk_document(
        [para(1, sentences)], document_key="k", chunk_size=20, overlap=0, count_tokens=count
    )
    joined = " ".join(c.text for c in chunks)
    assert joined.count("Sentence 5 ends.") == 1


def test_giant_paragraph_without_punctuation_is_split() -> None:
    blob = " ".join(f"x{i}" for i in range(1000))
    chunks = chunk_document(
        [para(1, blob)], document_key="k", chunk_size=100, overlap=10, count_tokens=count
    )
    assert len(chunks) >= 10
    assert all(c.token_count <= 100 for c in chunks)


def test_page_ranges_span_paragraph_pages() -> None:
    blocks = [para(3, "Results", True), para(3, words(10)), para(4, words(10, "z"))]
    (chunk,) = chunk_document(
        blocks, document_key="k", chunk_size=100, overlap=10, count_tokens=count
    )
    assert (chunk.page_start, chunk.page_end) == (3, 4)
    assert "\n\n" in chunk.text  # paragraph boundary preserved


def test_references_excluded_by_default() -> None:
    blocks = [
        para(1, "Intro", True),
        para(1, "body"),
        para(2, "References", True),
        para(2, "[1] Someone. A paper. 2020."),
    ]
    default = chunk_document(blocks, document_key="k", chunk_size=50, overlap=5, count_tokens=count)
    included = chunk_document(
        blocks,
        document_key="k",
        chunk_size=50,
        overlap=5,
        count_tokens=count,
        include_references=True,
    )
    assert [c.section for c in default] == ["Intro"]
    assert [c.section for c in included] == ["Intro", "References"]


def test_ids_are_deterministic_and_unique() -> None:
    blocks = [para(1, words(300))]
    a = chunk_document(blocks, document_key="hash1", chunk_size=50, overlap=5, count_tokens=count)
    b = chunk_document(blocks, document_key="hash1", chunk_size=50, overlap=5, count_tokens=count)
    c = chunk_document(blocks, document_key="hash2", chunk_size=50, overlap=5, count_tokens=count)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert len({x.chunk_id for x in a}) == len(a)
    assert a[0].chunk_id != c[0].chunk_id
    assert a[0].chunk_id == chunk_id_for("hash1", 0)
    assert [x.chunk_index for x in a] == list(range(len(a)))


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_document([], document_key="k", chunk_size=10, overlap=10)


def test_end_to_end_on_generated_paper() -> None:
    doc = load_pdf(make_paper_pdf())
    chunks = chunk_document(
        clean_blocks(doc.blocks, doc.page_count),
        document_key="h",
        chunk_size=120,
        overlap=20,
        count_tokens=word_count_tokens,
    )
    sections = [c.section for c in chunks]
    assert sections[0] == FRONT_MATTER
    assert {"Abstract", "1 Introduction", "2 Method", "3 Results", "4 Limitations"} <= set(sections)
    assert "References" not in sections
    limitations = next(c for c in chunks if c.section == "4 Limitations")
    assert limitations.page_start == 3
    assert "only English papers" in limitations.text
