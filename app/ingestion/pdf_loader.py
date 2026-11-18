"""PDF text extraction with PyMuPDF, preserving page numbers and typographic cues.

Output is a flat list of ``TextBlock`` segments in reading order. Each keeps its 1-based page
number, font size, boldness and position, which the cleaner (headers/footers) and chunker
(section headings) use. Content-stream order is used rather than geometric sorting, because
geometric sorting interleaves the columns of two-column papers while LaTeX-produced PDFs
normally emit text column by column.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from app.ingestion.errors import PermanentIngestionError

BOLD_FLAG = 1 << 4

KNOWN_SECTIONS = frozenset(
    {
        "abstract",
        "introduction",
        "related work",
        "related works",
        "background",
        "preliminaries",
        "method",
        "methods",
        "methodology",
        "approach",
        "proposed method",
        "model",
        "experiments",
        "experimental setup",
        "experimental results",
        "evaluation",
        "results",
        "discussion",
        "analysis",
        "conclusion",
        "conclusions",
        "conclusion and future work",
        "limitations",
        "future work",
        "ethics statement",
        "broader impact",
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "references",
        "bibliography",
        "appendix",
    }
)
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|[IVX]{1,6}\.|[A-H](?:\.\d+)*\.?)\s+\S")
_NUMBER_PREFIX = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|[IVX]{1,6}\.|[A-H](?:\.\d+)*\.?)\s+")
_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+[A-Za-z0-9])")
_ARXIV = re.compile(r"arXiv:\s?(\d{4})\.(\d{4,5})(v\d+)?", re.I)
_PDF_DATE_YEAR = re.compile(r"^D?:?((?:19|20)\d{2})")
# A line that is only a section number, e.g. "2.1" or "A.3": templates often emit it separately.
_NUMBER_ONLY = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|[IVX]{1,6}\.|[A-H](?:\.\d+)+\.?)$")


@dataclass(frozen=True)
class TextBlock:
    page_number: int  # 1-based
    text: str  # raw text; lines joined with "\n"
    font_size: float
    is_bold: bool
    y0: float
    y1: float
    page_height: float
    is_heading: bool = False
    is_title: bool = False


@dataclass
class PaperMetadata:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    page_count: int
    blocks: list[TextBlock]
    metadata: PaperMetadata
    body_font_size: float


def normalise_heading(text: str) -> str:
    """'3.1  Related Work:' -> 'related work' (for matching against known section names)."""
    text = _NUMBER_PREFIX.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip(" :.").lower()


def is_heading_text(text: str, *, font_size: float, is_bold: bool, body_size: float) -> bool:
    stripped = " ".join(text.split())
    words = stripped.split()
    if not stripped or len(stripped) > 120 or len(words) > 15:
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    if stripped.endswith((".", ",", ";")) and not _NUMBERED_HEADING.match(stripped + " x"):
        return False
    known = normalise_heading(stripped) in KNOWN_SECTIONS
    larger = font_size >= body_size * 1.15
    styled = larger or is_bold
    if known and (styled or stripped.isupper()):
        return True
    if styled and _NUMBERED_HEADING.match(stripped) and stripped[0].isalnum():
        # Avoid treating bold numbered list items ("1 We propose ...") as headings.
        return len(words) <= 10
    return font_size >= body_size * 1.3 and len(words) <= 12


def _line_style(line: dict) -> tuple[str, float, bool]:
    text = "".join(span["text"] for span in line["spans"])
    sizes: Counter[float] = Counter()
    bold_chars = 0
    total = 0
    for span in line["spans"]:
        n = len(span["text"].strip())
        if not n:
            continue
        sizes[round(span["size"], 1)] += n
        total += n
        if span["flags"] & BOLD_FLAG or "bold" in span["font"].lower():
            bold_chars += n
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    return text, size, total > 0 and bold_chars / total >= 0.6


def _body_font_size(pages: list[dict]) -> float:
    sizes: Counter[float] = Counter()
    for page in pages:
        for block in page["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


_Line = tuple[str, float, bool, tuple[float, float, float, float]]


def _make_block(
    run: list[_Line], page_number: int, height: float, is_heading: bool
) -> TextBlock | None:
    text = "\n".join(r[0] for r in run).strip()
    if not text:
        return None
    return TextBlock(
        page_number=page_number,
        text=text,
        font_size=max(r[1] for r in run),
        is_bold=all(r[2] for r in run),
        y0=min(r[3][1] for r in run),
        y1=max(r[3][3] for r in run),
        page_height=height,
        is_heading=is_heading,
    )


def _segments(page_number: int, page: dict, body_size: float) -> list[TextBlock]:
    """Split each PyMuPDF block into runs of lines with the same heading/body role."""
    out: list[TextBlock] = []
    height = page["height"]
    for block in page["blocks"]:
        if block.get("type") != 0:  # images
            continue
        runs: list[tuple[list[_Line], bool]] = []
        pending_number: str | None = None
        for line in block.get("lines", []):
            if abs(line["dir"][1]) > 0.01:  # rotated text, e.g. arXiv margin stamps
                continue
            text, size, bold = _line_style(line)
            if not text.strip():
                continue
            if _NUMBER_ONLY.match(text.strip()) and (bold or size >= body_size * 1.15):
                pending_number = text.strip()  # "2.1" on its own line: prefix the next line
                continue
            if pending_number is not None:
                text = f"{pending_number} {text.strip()}"
                pending_number = None
            heading = is_heading_text(text, font_size=size, is_bold=bold, body_size=body_size)
            # A heading line always starts a new run, and so does the first line after one.
            if not runs or heading or runs[-1][1]:
                runs.append(([], heading))
            runs[-1][0].append((text, size, bold, tuple(line["bbox"])))
        for lines, is_heading in runs:
            made = _make_block(lines, page_number, height, is_heading)
            if made is not None:
                out.append(made)
    return out


def merge_wrapped_headings(blocks: list[TextBlock]) -> list[TextBlock]:
    """Join a heading that wrapped onto several lines/blocks ('Appendices for X for' + 'Y').

    A following heading is treated as a continuation only if it is on the same page, has the
    same font size, and does not itself look like the start of a new section.
    """
    merged: list[TextBlock] = []
    for b in blocks:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.is_heading
            and b.is_heading
            and prev.page_number == b.page_number
            and abs(prev.font_size - b.font_size) < 0.5
            and not _NUMBERED_HEADING.match(b.text)
            and normalise_heading(b.text) not in KNOWN_SECTIONS
        ):
            merged[-1] = TextBlock(**{**prev.__dict__, "text": f"{prev.text} {b.text}", "y1": b.y1})
        else:
            merged.append(b)
    return merged


def _mark_title(blocks: list[TextBlock]) -> list[TextBlock]:
    """The largest text in the top half of page 1 is the title, not a section heading."""
    first = [b for b in blocks if b.page_number == 1 and b.y0 < b.page_height / 2]
    if not first:
        return blocks
    biggest = max(b.font_size for b in first)
    body = sorted(b.font_size for b in blocks)[len(blocks) // 2]
    if biggest < body * 1.2:
        return blocks
    result = []
    for b in blocks:
        if b in first and b.font_size == biggest:
            b = TextBlock(**{**b.__dict__, "is_title": True, "is_heading": False})
        result.append(b)
    return result


def _clean_meta_title(title: str | None) -> str | None:
    if not title:
        return None
    title = " ".join(title.split())
    lowered = title.lower()
    junk = ("microsoft word", "untitled", ".pdf", ".doc", ".tex", "arxiv")
    if len(title) < 4 or any(j in lowered for j in junk):
        return None
    return title


def _split_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"\s*(?:;|,|\band\b|&)\s*", raw)
    return [p.strip() for p in parts if len(p.strip()) > 1][:50]


def extract_metadata(
    raw_meta: dict[str, str], blocks: list[TextBlock], front_text: str | None = None
) -> PaperMetadata:
    raw = {k: v for k, v in (raw_meta or {}).items() if isinstance(v, str) and v}
    title = _clean_meta_title(raw.get("title"))
    if title is None:
        title_blocks = [b for b in blocks if b.is_title]
        if title_blocks:
            title = " ".join(" ".join(b.text.split()) for b in title_blocks)[:500] or None

    if front_text is None:
        front_text = "\n".join(b.text for b in blocks if b.page_number <= 2)
    doi_match = _DOI.search(front_text)
    arxiv_match = _ARXIV.search(front_text) or _ARXIV.search(raw.get("subject", ""))
    arxiv_id = None
    year = None
    if arxiv_match:
        arxiv_id = f"{arxiv_match.group(1)}.{arxiv_match.group(2)}"
        year = 2000 + int(arxiv_match.group(1)[:2])
    if year is None:
        for key in ("creationDate", "modDate"):
            m = _PDF_DATE_YEAR.match(raw.get(key, ""))
            if m:
                year = int(m.group(1))
                break
    return PaperMetadata(
        title=title,
        authors=_split_authors(raw.get("author")),
        year=year,
        doi=doi_match.group(1).rstrip(".") if doi_match else None,
        arxiv_id=arxiv_id,
        raw=raw,
    )


def load_pdf(data: bytes) -> ExtractedDocument:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PermanentIngestionError("PDF could not be opened") from exc
    with doc:
        pages = [page.get_text("dict") for page in doc]
        body = _body_font_size(pages)
        blocks: list[TextBlock] = []
        for number, page in enumerate(pages, start=1):
            blocks.extend(_segments(number, page, body))
        blocks = merge_wrapped_headings(_mark_title(blocks)) if blocks else blocks
        # Raw text of the first pages (incl. rotated margin stamps) for DOI/arXiv detection.
        front_text = "\n".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        return ExtractedDocument(
            page_count=doc.page_count,
            blocks=blocks,
            metadata=extract_metadata(doc.metadata or {}, blocks, front_text),
            body_font_size=body,
        )
