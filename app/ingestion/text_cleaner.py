"""Text normalisation and removal of running headers/footers."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import replace

from app.ingestion.pdf_loader import TextBlock

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏﻿]")
_SPACES = re.compile(r"[ \t ]+")
MARGIN_FRACTION = 0.08


def normalize_text(text: str) -> str:
    """NFKC (splits ligatures like 'ﬁ'), drop control chars, re-join hyphenated line breaks
    when the next line continues in lower case, and collapse whitespace into single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(
        lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0).replace("\n", ""),
        text,
    )
    text = text.replace("\n", " ")
    return _SPACES.sub(" ", text).strip()


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def _in_margin(block: TextBlock) -> bool:
    return block.y1 <= block.page_height * MARGIN_FRACTION or block.y0 >= block.page_height * (
        1 - MARGIN_FRACTION
    )


def remove_headers_and_footers(blocks: list[TextBlock], page_count: int) -> list[TextBlock]:
    """Drop margin text that repeats on at least half the pages, and bare page numbers."""
    pages_by_key: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        if _in_margin(b):
            pages_by_key[_margin_key(b.text)].add(b.page_number)
    threshold = max(2, (page_count + 1) // 2)
    repeated = {k for k, pages in pages_by_key.items() if len(pages) >= threshold}

    kept = []
    for b in blocks:
        if _in_margin(b):
            key = _margin_key(b.text)
            if key in repeated or re.fullmatch(r"(page )?#( of #)?", key):
                continue
        kept.append(b)
    return kept


def clean_blocks(blocks: list[TextBlock], page_count: int) -> list[TextBlock]:
    cleaned = []
    for b in remove_headers_and_footers(blocks, page_count):
        text = normalize_text(b.text)
        if text:
            cleaned.append(replace(b, text=text))
    return cleaned
