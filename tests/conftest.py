"""Shared pytest fixtures for the FDS Reconciler test suite."""
from __future__ import annotations

import pytest

from app.domain.chat import ChatAnswer, Citation
from app.domain.chunk import Chunk
from app.domain.comparison import ComparisonResult, DiffEntry, MatchEntry, MissingEntry
from app.domain.section import Location, ParsedDoc, Section, TableData
from app.domain.summary import ExecutiveSummary, ImportantChange
from app.domain.verdict import MissingExplanationBatch, PairwiseVerdict
from tests.fixtures.parsed_docs import make_parsed_doc_a, make_parsed_doc_b, make_section


@pytest.fixture
def parsed_doc_a() -> ParsedDoc:
    return make_parsed_doc_a()


@pytest.fixture
def parsed_doc_b() -> ParsedDoc:
    return make_parsed_doc_b()


@pytest.fixture
def sample_section_a() -> Section:
    return make_section(
        "A", "Process Stages", "3.1",
        body_text="Describes the three processing stages.",
        page_number=4,
    )


@pytest.fixture
def sample_section_b() -> Section:
    return make_section(
        "B", "Process Stages", "3.1",
        body_text="Describes the three processing stages. Now expanded.",
        filename="FDS_V5.docx",
    )


@pytest.fixture
def aligned_pair(sample_section_a: Section, sample_section_b: Section) -> tuple[Section, Section, float]:
    return sample_section_a, sample_section_b, 0.95


@pytest.fixture
def chunk(sample_section_a: Section) -> Chunk:
    return Chunk(
        id=f"{sample_section_a.id}::chunk-0",
        section_id=sample_section_a.id,
        doc_id="A",
        chunk_type="prose",
        text="3.1 > Process Stages\n\nDescribes the three processing stages.",
        display_text="Describes the three processing stages.",
        location=sample_section_a.location,
    )


@pytest.fixture
def comparison_result() -> ComparisonResult:
    return ComparisonResult(
        match=[
            MatchEntry(
                textA="Stage 1 is intake.",
                textB="Stage 1 is intake.",
                source="FDS_V0.pdf · §3.1 · page 4 + FDS_V5.docx · §3.1",
            ),
        ],
        diff=[
            DiffEntry(
                docA_text="Base price is determined by tier and volume.",
                docB_text="Base price is determined by tier and volume. Discounts now apply.",
                reason="V5 adds discount logic.",
                sourceA="FDS_V0.pdf · §3.2 · page 5",
                sourceB="FDS_V5.docx · §3.2",
            ),
        ],
        missing=[
            MissingEntry(
                text="Legacy records are migrated in batch.",
                source_file="FDS_V0.pdf",
                location="FDS_V0.pdf · §4.3 · page 9",
                explanation="Section present in V0 but absent from V5.",
            ),
        ],
    )
