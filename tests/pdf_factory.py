"""Builds small but realistic research-paper PDFs for tests (no binary fixtures in git)."""

from __future__ import annotations

import pymupdf

LOREM = (
    "Retrieval-augmented generation combines a parametric language model with a non-parametric "
    "memory of documents. The retriever selects passages that are relevant to the input and the "
    "generator conditions on them. This design improves factual accuracy on knowledge-intensive "
    "tasks while keeping the model size fixed. "
)

DEFAULT_SECTIONS: list[tuple[str, str]] = [
    ("Abstract", "We study retrieval for research papers. " + LOREM),
    ("1 Introduction", LOREM * 3),
    (
        "2 Method",
        "Our method uses dense and sparse retrieval with reciprocal rank fusion. " + LOREM * 2,
    ),
    ("3 Results", "Hybrid retrieval improved recall at ten by a wide margin. " + LOREM * 2),
    ("4 Limitations", "The evaluation covers only English papers and a small corpus. " + LOREM),
    (
        "References",
        "[1] P. Lewis et al. Retrieval-augmented generation. NeurIPS 2020.\n"
        "[2] V. Karpukhin et al. Dense passage retrieval. EMNLP 2020.",
    ),
]


def make_paper_pdf(
    *,
    title: str = "Hybrid Retrieval for Scientific Question Answering",
    authors: str = "Alice Smith, Bob Jones",
    sections: list[tuple[str, str]] | None = None,
    doi: str | None = "10.1234/rp.2023.0042",
    metadata_title: str | None = None,
    creation_date: str = "D:20230115120000",
    running_header: str = "Preprint. Under review.",
    sections_per_page: int = 2,
) -> bytes:
    sections = DEFAULT_SECTIONS if sections is None else sections
    doc = pymupdf.open()
    width, height = 612, 792
    page_index = 0

    def new_page() -> pymupdf.Page:
        nonlocal page_index
        page_index += 1
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 30), running_header, fontsize=8, fontname="helv")
        page.insert_text((300, 770), str(page_index), fontsize=8, fontname="helv")
        return page

    page = new_page()
    page.insert_text((72, 90), title, fontsize=18, fontname="hebo")
    page.insert_text((72, 115), authors, fontsize=11, fontname="helv")
    y = 140.0
    if doi:
        page.insert_text((72, y), f"doi: {doi}", fontsize=9, fontname="helv")
        y += 25

    for i, (heading, body) in enumerate(sections):
        if i and i % sections_per_page == 0:
            page = new_page()
            y = 70.0
        page.insert_text((72, y), heading, fontsize=12, fontname="hebo")
        y += 10
        rect = pymupdf.Rect(72, y, width - 72, height - 60)
        leftover = page.insert_textbox(rect, body, fontsize=10, fontname="tiro")
        if leftover < 0:
            raise ValueError(f"section {heading!r} does not fit on one page")
        used = rect.height - leftover
        y += used + 25

    doc.set_metadata(
        {
            "title": metadata_title or "",
            "author": authors,
            "creationDate": creation_date,
        }
    )
    data = doc.tobytes()
    doc.close()
    return data


def make_blank_pdf(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    return data
