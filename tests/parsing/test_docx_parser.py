"""Tests for parse_docx() — runs against samples/FDS_PriceBook_V5.docx when present."""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_PDF = Path(__file__).parent.parent.parent / "samples" / "FDS_PriceBook_V0.pdf"
SAMPLE_DOCX = Path(__file__).parent.parent.parent / "samples" / "FDS_PriceBook_V5.docx"
pytestmark = pytest.mark.skipif(
    not SAMPLE_DOCX.exists(),
    reason="samples/FDS_PriceBook_V5.docx not present",
)


@pytest.fixture(scope="module")
def parsed():
    from app.parsing.docx_parser import parse_docx
    return parse_docx(SAMPLE_DOCX, "B")


def test_parse_docx_returns_correct_doc_id(parsed) -> None:
    assert parsed.doc_id == "B"


def test_parse_docx_page_number_is_none_for_all_sections(parsed) -> None:
    for sec in parsed.sections:
        assert sec.location.page_number is None, (
            f"Section {sec.id} has page_number={sec.location.page_number}; DOCX has no fixed pages"
        )


def test_parse_docx_heading_levels_detected(parsed) -> None:
    depths = [len(sec.location.heading_path) for sec in parsed.sections]
    assert max(depths) >= 2, "Expected headings at multiple depth levels"


def test_parse_docx_at_least_one_table_extracted(parsed) -> None:
    tables = [t for sec in parsed.sections for t in sec.tables]
    assert len(tables) >= 1, "No tables extracted from DOCX"


def test_parse_docx_section_count_snapshot(parsed) -> None:
    count = len(parsed.sections)
    assert 10 <= count <= 100, f"Unexpected section count: {count}"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="PDF sample not present for comparison")
def test_parse_docx_has_more_sections_than_pdf(parsed) -> None:
    from app.parsing.pdf_parser import parse_pdf
    pdf_parsed = parse_pdf(SAMPLE_PDF, "A")
    assert len(parsed.sections) >= len(pdf_parsed.sections), (
        "V5 DOCX should have at least as many sections as V0 PDF"
    )


def test_parse_docx_filename_stored_correctly(parsed) -> None:
    assert parsed.filename == SAMPLE_DOCX.name
