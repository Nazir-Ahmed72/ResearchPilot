"""PDF validation, extraction, cleaning and structure detection."""

from __future__ import annotations

import pytest

from app.ingestion.errors import PdfValidationError
from app.ingestion.pdf_loader import TextBlock, is_heading_text, load_pdf, normalise_heading
from app.ingestion.text_cleaner import clean_blocks, normalize_text, remove_headers_and_footers
from app.ingestion.validation import sanitize_filename, validate_pdf
from tests.pdf_factory import make_blank_pdf, make_encrypted_pdf, make_paper_pdf

MB = 1024 * 1024


def _validate(data: bytes, filename: str = "paper.pdf", content_type: str = "application/pdf"):
    return validate_pdf(
        data, filename=filename, content_type=content_type, max_size_bytes=5 * MB, max_pages=50
    )


# --- validation -------------------------------------------------------------------------------


def test_valid_pdf_passes() -> None:
    result = _validate(make_paper_pdf())
    assert result.page_count == 3
    assert result.filename == "paper.pdf"


@pytest.mark.parametrize(
    ("data", "filename", "content_type", "code"),
    [
        (b"%PDF-1.7 fake", "notes.txt", "application/pdf", "unsupported_media_type"),
        (b"%PDF-1.7 fake", "paper.pdf", "image/png", "unsupported_media_type"),
        (b"", "paper.pdf", "application/pdf", "invalid_pdf"),
        (b"MZ\x90\x00 this is an exe", "paper.pdf", "application/pdf", "invalid_pdf"),
        (b"%PDF-1.7\n garbage that is not a pdf", "paper.pdf", "application/pdf", "invalid_pdf"),
    ],
)
def test_invalid_uploads_rejected(data: bytes, filename: str, content_type: str, code: str) -> None:
    with pytest.raises(PdfValidationError) as err:
        _validate(data, filename, content_type)
    assert err.value.code == code


def test_oversized_file_rejected() -> None:
    with pytest.raises(PdfValidationError) as err:
        validate_pdf(
            make_paper_pdf(),
            filename="a.pdf",
            content_type="application/pdf",
            max_size_bytes=100,
            max_pages=50,
        )
    assert err.value.code == "payload_too_large"


def test_too_many_pages_rejected() -> None:
    with pytest.raises(PdfValidationError, match="limit is 2"):
        validate_pdf(
            make_blank_pdf(3),
            filename="a.pdf",
            content_type="application/pdf",
            max_size_bytes=MB,
            max_pages=2,
        )


def test_encrypted_pdf_rejected() -> None:
    with pytest.raises(PdfValidationError, match="Encrypted"):
        _validate(make_encrypted_pdf())


def test_octet_stream_content_type_accepted() -> None:
    assert _validate(make_paper_pdf(), content_type="application/octet-stream").page_count == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Windows\\system32\\evil.pdf", "evil.pdf"),
        ("bad\x00name\n.pdf", "badname.pdf"),
        ("", "upload.pdf"),
        (None, "upload.pdf"),
        ("x" * 300 + ".pdf", "x" * 255),
    ],
)
def test_filename_sanitisation(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


# --- extraction -------------------------------------------------------------------------------


def test_extraction_preserves_pages_and_metadata() -> None:
    doc = load_pdf(make_paper_pdf())
    assert doc.page_count == 3
    assert {b.page_number for b in doc.blocks} == {1, 2, 3}
    meta = doc.metadata
    assert meta.title == "Hybrid Retrieval for Scientific Question Answering"  # from page 1 font
    assert meta.authors == ["Alice Smith", "Bob Jones"]
    assert meta.year == 2023
    assert meta.doi == "10.1234/rp.2023.0042"


def test_metadata_title_preferred_when_meaningful() -> None:
    doc = load_pdf(make_paper_pdf(metadata_title="Official Title From Metadata"))
    assert doc.metadata.title == "Official Title From Metadata"


def test_junk_metadata_title_ignored() -> None:
    doc = load_pdf(make_paper_pdf(metadata_title="Microsoft Word - draft3.docx"))
    assert doc.metadata.title == "Hybrid Retrieval for Scientific Question Answering"


def test_headings_detected_and_title_not_a_heading() -> None:
    doc = load_pdf(make_paper_pdf())
    headings = [b.text for b in doc.blocks if b.is_heading]
    assert headings == [
        "Abstract",
        "1 Introduction",
        "2 Method",
        "3 Results",
        "4 Limitations",
        "References",
    ]
    assert any(b.is_title for b in doc.blocks)


@pytest.mark.parametrize(
    ("text", "size", "bold", "expected"),
    [
        ("3.1 Related Work", 10, True, True),
        ("REFERENCES", 10, False, True),
        ("Conclusion", 10, True, True),
        ("1 We propose a new method that does many things well.", 10, True, False),
        ("This is an ordinary body sentence.", 10, False, False),
        ("Introduction", 10, False, False),  # known name but unstyled, mixed case
        ("A Survey", 14, False, True),  # much larger font
        ("42", 14, True, False),
    ],
)
def test_heading_heuristics(text: str, size: float, bold: bool, expected: bool) -> None:
    assert is_heading_text(text, font_size=size, is_bold=bold, body_size=10) is expected


def test_normalise_heading() -> None:
    assert normalise_heading("3.1  Related Work:") == "related work"
    assert normalise_heading("IV. CONCLUSION") == "conclusion"


# --- cleaning ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("retrie-\nval augmented", "retrieval augmented"),
        ("state-of-the-\nArt", "state-of-the-Art"),
        ("the ﬁrst eﬃcient", "the first efficient"),
        ("line one\nline two", "line one line two"),
        ("tab\tand  spaces\u00a0here", "tab and spaces here"),
        ("zero\u200bwidth\x07bell", "zerowidthbell"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def _block(page: int, text: str, y0: float, y1: float) -> TextBlock:
    return TextBlock(page, text, 10, False, y0, y1, 792)


def test_running_headers_and_page_numbers_removed() -> None:
    blocks = []
    for p in range(1, 5):
        blocks += [
            _block(p, "Preprint. Under review.", 20, 32),
            _block(p, f"Body text on page {p}", 300, 320),
            _block(p, str(p), 765, 775),
        ]
    kept = remove_headers_and_footers(blocks, page_count=4)
    assert [b.text for b in kept] == [f"Body text on page {p}" for p in range(1, 5)]


def test_unique_margin_text_is_kept() -> None:
    blocks = [_block(1, "Footnote: only once", 760, 775), _block(2, "body", 300, 310)]
    assert len(remove_headers_and_footers(blocks, page_count=2)) == 2


def test_clean_blocks_on_real_pdf_drops_running_header() -> None:
    doc = load_pdf(make_paper_pdf())
    cleaned = clean_blocks(doc.blocks, doc.page_count)
    assert not any("Under review" in b.text for b in cleaned)
    assert all("\n" not in b.text for b in cleaned)


# --- regressions found on a real arXiv paper (2005.11401) ----------------------------------


def _line(text: str, size: float = 10.0, bold: bool = True, y: float = 100.0) -> dict:
    flags = 16 if bold else 0
    return {
        "dir": (1.0, 0.0),
        "bbox": (72.0, y, 300.0, y + size),
        "spans": [{"text": text, "size": size, "flags": flags, "font": "Times"}],
    }


def test_separate_section_number_line_is_glued_to_heading() -> None:
    from app.ingestion.pdf_loader import _segments

    page = {
        "height": 792,
        "blocks": [
            {"type": 0, "lines": [_line("2.1", y=100), _line("Models", y=100)]},
            {"type": 0, "lines": [_line("Body text follows here.", bold=False, y=120)]},
        ],
    }
    blocks = _segments(3, page, body_size=10.0)
    assert blocks[0].text == "2.1 Models" and blocks[0].is_heading
    assert not blocks[1].is_heading


def test_unstyled_number_line_is_not_glued() -> None:
    from app.ingestion.pdf_loader import _segments

    page = {
        "height": 792,
        "blocks": [
            {
                "type": 0,
                "lines": [_line("42", bold=False), _line("apples were counted.", bold=False)],
            }
        ],
    }
    (block,) = _segments(1, page, body_size=10.0)
    assert block.text == "42\napples were counted." and not block.is_heading


def test_wrapped_heading_lines_are_merged() -> None:
    from app.ingestion.pdf_loader import merge_wrapped_headings

    def h(text: str, page: int = 17) -> TextBlock:
        return TextBlock(page, text, 14, True, 100, 110, 792, is_heading=True)

    merged = merge_wrapped_headings(
        [
            h("Appendices for Retrieval-Augmented Generation for"),
            h("Knowledge-Intensive NLP Tasks"),
            h("A Implementation Details"),  # new numbered section: not merged
            h("References"),
        ]  # known section name: not merged
    )
    assert [b.text for b in merged] == [
        "Appendices for Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "A Implementation Details",
        "References",
    ]


def test_arxiv_id_and_year_from_raw_front_text() -> None:
    from app.ingestion.pdf_loader import extract_metadata

    meta = extract_metadata(
        {"creationDate": "D:20210412000000"},
        [],
        front_text="arXiv:2005.11401v4  [cs.CL]  12 Apr 2021",
    )
    assert meta.arxiv_id == "2005.11401"
    assert meta.year == 2020  # submission year from the ID, not the v4 PDF date
