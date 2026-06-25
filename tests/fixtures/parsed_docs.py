"""Reusable ParsedDoc/Section builders for tests."""
from __future__ import annotations

from app.domain.section import Location, ParsedDoc, Section, TableData


def make_location(
    filename: str,
    heading_number: str | None = None,
    heading_path: list[str] | None = None,
    page_number: int | None = None,
) -> Location:
    return Location(
        filename=filename,
        heading_number=heading_number,
        heading_path=heading_path or ([heading_number] if heading_number else []),
        page_number=page_number,
    )


def make_section(
    doc_id: str,
    heading: str,
    heading_number: str | None = None,
    body_text: str = "",
    tables: list[TableData] | None = None,
    bullets: list[str] | None = None,
    filename: str | None = None,
    page_number: int | None = None,
) -> Section:
    fname = filename or f"{doc_id}.pdf"
    if heading_number:
        section_id = f"{doc_id}::{heading_number}"
        path = [heading_number, heading]
    else:
        slug = heading.lower().replace(" ", "-")[:20]
        section_id = f"{doc_id}::{slug}"
        path = [heading]
    return Section(
        id=section_id,
        location=Location(
            filename=fname,
            heading_number=heading_number,
            heading_path=path,
            page_number=page_number,
        ),
        heading=heading,
        body_text=body_text,
        tables=tables or [],
        bullets=bullets or [],
    )


# ── small realistic ParsedDocs used as fixtures ────────────────────────────────

def make_parsed_doc_a() -> ParsedDoc:
    """V0-style doc with 5 sections; 4 have counterparts in doc B, 1 is unique."""
    sections = [
        make_section("A", "Process Stages",    "3.1",
                     body_text="Describes the three processing stages.",
                     page_number=4),
        make_section("A", "Price Calculation", "3.2",
                     body_text="Base price is determined by tier and volume.",
                     page_number=5),
        make_section("A", "Output Format",     "3.3",
                     body_text="Results are emitted as JSON.",
                     page_number=6),
        make_section("A", "Phase A Live QA",   "4.1",
                     body_text="Live QA runs against production data.",
                     page_number=7),
        make_section("A", "Phase B Testing",   "4.2",
                     body_text="Regression suite runs nightly.",
                     page_number=8),
        # Unique to A (no counterpart in B)
        make_section("A", "Legacy Migration",  "4.3",
                     body_text="Legacy records are migrated in batch.",
                     page_number=9),
    ]
    return ParsedDoc(doc_id="A", filename="FDS_V0.pdf", sections=sections)


def make_parsed_doc_b() -> ParsedDoc:
    """V5-style doc with 6 sections; 5 align with doc A, 1 is new."""
    sections = [
        make_section("B", "Process Stages",    "3.1",
                     body_text="Describes the three processing stages.",
                     filename="FDS_V5.docx"),
        make_section("B", "Price Calculation", "3.2",
                     body_text="Base price is determined by tier and volume. Discounts now apply.",
                     filename="FDS_V5.docx"),
        make_section("B", "Output Format",     "3.3",
                     body_text="Results are emitted as JSON.",
                     filename="FDS_V5.docx"),
        make_section("B", "Phase A Live QA",   "4.1",
                     body_text="Live QA runs against production data.",
                     filename="FDS_V5.docx"),
        make_section("B", "Phase B Testing",   "4.2",
                     body_text="Regression suite runs nightly.",
                     filename="FDS_V5.docx"),
        # New in B (no counterpart in A)
        make_section("B", "Integration Checks", "5.1",
                     body_text="New integration test layer added in V5.",
                     filename="FDS_V5.docx"),
    ]
    return ParsedDoc(doc_id="B", filename="FDS_V5.docx", sections=sections)
