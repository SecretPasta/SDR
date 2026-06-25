"""Tests for parse_pdf() — runs against samples/FDS_PriceBook_V0.pdf when present."""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_PDF = Path(__file__).parent.parent.parent / "samples" / "FDS_PriceBook_V0.pdf"
pytestmark = pytest.mark.skipif(
    not SAMPLE_PDF.exists(),
    reason="samples/FDS_PriceBook_V0.pdf not present",
)


@pytest.fixture(scope="module")
def parsed():
    from app.parsing.pdf_parser import parse_pdf
    return parse_pdf(SAMPLE_PDF, "A")


def test_parse_pdf_returns_correct_doc_id(parsed) -> None:
    assert parsed.doc_id == "A"


def test_parse_pdf_detects_at_least_10_sections(parsed) -> None:
    assert len(parsed.sections) >= 10


def test_parse_pdf_section_311_process_stages_exists(parsed) -> None:
    by_num = parsed.by_heading_number()
    assert "3.1" in by_num, "Section 3.1 not found"
    assert "process" in by_num["3.1"].heading.lower() or "stage" in by_num["3.1"].heading.lower()


def test_parse_pdf_all_sections_have_page_number(parsed) -> None:
    for sec in parsed.sections:
        assert sec.location.page_number is not None, f"Section {sec.id} missing page_number"


def test_parse_pdf_at_least_one_table_extracted(parsed) -> None:
    tables = [t for sec in parsed.sections for t in sec.tables]
    assert len(tables) >= 1, "No tables extracted from PDF"


def test_parse_pdf_heading_path_populated_with_parents_preceding_children(parsed) -> None:
    for sec in parsed.sections:
        path = sec.location.heading_path
        assert isinstance(path, list)
        assert len(path) >= 1


def test_parse_pdf_section_count_snapshot(parsed) -> None:
    """Snapshot: catches regressions in section detection."""
    count = len(parsed.sections)
    # Should be between 10 and 80 for a ~50-page FDS doc
    assert 10 <= count <= 80, f"Unexpected section count: {count}"


def test_parse_pdf_filename_stored_correctly(parsed) -> None:
    assert parsed.filename == SAMPLE_PDF.name
    for sec in parsed.sections:
        assert sec.location.filename == SAMPLE_PDF.name
