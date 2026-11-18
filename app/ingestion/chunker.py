"""Structure-aware chunking.

1. Blocks are grouped into sections at detected headings.
2. Within a section, paragraphs are packed greedily up to ``chunk_size`` tokens. Paragraphs that
   are too long are split at sentence boundaries, and over-long sentences at word windows.
3. Consecutive chunks of the same section share ~``overlap`` tokens of trailing sentences so a
   statement spanning a boundary is retrievable from either side.
4. Chunks never cross section boundaries, so every chunk has exactly one section label, and each
   records the page range it came from.

Token counting is injected, so the chunker is model-agnostic and testable without a tokenizer.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from app.ingestion.pdf_loader import TextBlock, normalise_heading

TokenCounter = Callable[[str], int]
CHUNK_NAMESPACE = uuid.UUID("6f1c3c58-3c2e-4f57-9a0e-9b8a5e0f7a11")
FRONT_MATTER = "Front Matter"
REFERENCE_SECTIONS = frozenset({"references", "bibliography"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    chunk_index: int
    text: str
    section: str
    page_start: int
    page_end: int
    token_count: int


@dataclass
class Section:
    title: str
    paragraphs: list[tuple[int, str]]  # (page_number, text)


@dataclass(frozen=True)
class _Unit:
    page: int
    text: str
    tokens: int
    paragraph: int  # units from the same paragraph are joined with a space, else a blank line


def word_count_tokens(text: str) -> int:
    """Fallback counter (~1.3 tokens per word for English subword tokenizers)."""
    return max(1, round(len(text.split()) * 1.3))


def chunk_id_for(document_key: str, chunk_index: int) -> str:
    """Deterministic: re-ingesting the same file yields the same IDs (idempotent upserts)."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{document_key}:{chunk_index}"))


def build_sections(blocks: list[TextBlock]) -> list[Section]:
    sections: list[Section] = [Section(FRONT_MATTER, [])]
    for block in blocks:
        if block.is_heading:
            sections.append(Section(" ".join(block.text.split()), []))
        else:
            sections[-1].paragraphs.append((block.page_number, block.text))
    return [s for s in sections if s.paragraphs]


def _split_long(text: str, page: int, para: int, size: int, count: TokenCounter) -> list[_Unit]:
    units: list[_Unit] = []
    for sentence in _SENTENCE_END.split(text):
        n = count(sentence)
        if n <= size:
            units.append(_Unit(page, sentence, n, para))
            continue
        words = sentence.split()
        step = max(1, int(len(words) * size / n))  # proportional word windows
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + step])
            units.append(_Unit(page, piece, count(piece), para))
    return units


def _units(section: Section, size: int, count: TokenCounter) -> list[_Unit]:
    units: list[_Unit] = []
    for para_index, (page, text) in enumerate(section.paragraphs):
        n = count(text)
        if n <= size:
            units.append(_Unit(page, text, n, para_index))
        else:
            units.extend(_split_long(text, page, para_index, size, count))
    return units


def _join(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i:
            parts.append(" " if unit.paragraph == units[i - 1].paragraph else "\n\n")
        parts.append(unit.text)
    return "".join(parts)


def _pack(units: list[_Unit], size: int, overlap: int) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0
    fresh = 0  # units in `current` not carried over from the previous chunk
    for unit in units:
        if current and current_tokens + unit.tokens > size and fresh:
            groups.append(current)
            carry: list[_Unit] = []
            carried = 0
            for prev in reversed(current):
                if carried + prev.tokens > overlap or carried + prev.tokens + unit.tokens > size:
                    break
                carry.insert(0, prev)
                carried += prev.tokens
            current, current_tokens, fresh = carry, carried, 0
        current.append(unit)
        current_tokens += unit.tokens
        fresh += 1
    if current and fresh:
        groups.append(current)
    return groups


def chunk_document(
    blocks: list[TextBlock],
    *,
    document_key: str,
    chunk_size: int,
    overlap: int,
    count_tokens: TokenCounter = word_count_tokens,
    include_references: bool = False,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for section in build_sections(blocks):
        if not include_references and normalise_heading(section.title) in REFERENCE_SECTIONS:
            continue
        for group in _pack(_units(section, chunk_size, count_tokens), chunk_size, overlap):
            text = _join(group)
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(document_key, index),
                    chunk_index=index,
                    text=text,
                    section=section.title,
                    page_start=min(u.page for u in group),
                    page_end=max(u.page for u in group),
                    token_count=count_tokens(text),
                )
            )
    return chunks
