"""Tests for all domain model methods — serialisation, citations, markdown output."""
from __future__ import annotations

import pytest

from app.domain.chat import ChatAnswer, Citation
from app.domain.chunk import Chunk
from app.domain.comparison import ComparisonResult, DiffEntry, MatchEntry, MissingEntry
from app.domain.section import Location, Section, TableData


# ── TableData ─────────────────────────────────────────────────────────────────

class TestTableDataToMarkdown:
    def test_basic_table_produces_valid_markdown(self) -> None:
        t = TableData(headers=["Phase", "Status"], rows=[["A", "Live"], ["B", "Pending"]])
        md = t.to_markdown()
        assert "| Phase | Status |" in md
        assert "| --- | --- |" in md
        assert "| A | Live |" in md
        assert "| B | Pending |" in md

    def test_empty_table_returns_empty_string(self) -> None:
        assert TableData(headers=[], rows=[]).to_markdown() == ""

    def test_header_row_comes_before_separator(self) -> None:
        t = TableData(headers=["Col"], rows=[["val"]])
        lines = t.to_markdown().splitlines()
        assert lines[0].startswith("| Col")
        assert "---" in lines[1]
        assert lines[2].startswith("| val")


# ── Location.cite() ───────────────────────────────────────────────────────────

class TestLocationCite:
    def test_pdf_with_page_and_heading_number(self) -> None:
        loc = Location(filename="doc.pdf", page_number=5, heading_number="3.1",
                       heading_path=["3.1", "Process Stages"])
        assert loc.cite() == "doc.pdf · §3.1 · page 5"

    def test_pdf_with_page_no_heading_number_falls_back_to_path(self) -> None:
        loc = Location(filename="doc.pdf", page_number=3,
                       heading_path=["Appendix A"])
        assert loc.cite() == "doc.pdf · §Appendix A · page 3"

    def test_docx_with_heading_number_no_page(self) -> None:
        loc = Location(filename="doc.docx", heading_number="2.1")
        assert loc.cite() == "doc.docx · §2.1"

    def test_docx_with_neither_heading_number_nor_path(self) -> None:
        loc = Location(filename="doc.docx")
        assert loc.cite() == "doc.docx"


# ── Section.for_embedding() ───────────────────────────────────────────────────

class TestSectionForEmbedding:
    def test_prepends_full_heading_breadcrumb(self) -> None:
        sec = Section(
            id="A::3.1",
            location=Location(filename="a.pdf", heading_path=["3. Features", "3.1 Stages"]),
            heading="Stages",
            body_text="Some body text.",
        )
        result = sec.for_embedding()
        assert result.startswith("3. Features > 3.1 Stages")
        assert "Some body text." in result

    def test_no_breadcrumb_when_heading_path_empty(self) -> None:
        sec = Section(
            id="A::x",
            location=Location(filename="a.pdf", heading_path=[]),
            heading="Title",
            body_text="Body.",
        )
        result = sec.for_embedding()
        assert result == "Body."

    def test_bullets_included_in_embedding_text(self) -> None:
        sec = Section(
            id="A::x",
            location=Location(filename="a.pdf", heading_path=["H"]),
            heading="H",
            bullets=["item one", "item two"],
        )
        result = sec.for_embedding()
        assert "item one" in result
        assert "item two" in result


# ── Chunk ─────────────────────────────────────────────────────────────────────

class TestChunk:
    def _make_chunk(self, seq: int = 0) -> Chunk:
        loc = Location(filename="doc.pdf", heading_number="3.1",
                       heading_path=["3.1", "Process Stages"], page_number=4)
        return Chunk(
            id=f"A::3.1::chunk-{seq}",
            section_id="A::3.1",
            doc_id="A",
            chunk_type="prose",
            text="3.1 > Process Stages\n\nBody text here.",
            display_text="Body text here.",
            location=loc,
        )

    def test_id_follows_section_id_chunk_seq_pattern(self) -> None:
        chunk = self._make_chunk(seq=2)
        assert chunk.id == "A::3.1::chunk-2"

    def test_index_metadata_includes_display_text(self) -> None:
        meta = self._make_chunk().index_metadata()
        assert meta["display_text"] == "Body text here."

    def test_index_metadata_includes_doc_id_and_section_id(self) -> None:
        meta = self._make_chunk().index_metadata()
        assert meta["doc_id"] == "A"
        assert meta["section_id"] == "A::3.1"

    def test_index_metadata_includes_heading_path_as_list(self) -> None:
        meta = self._make_chunk().index_metadata()
        assert isinstance(meta["heading_path"], list)
        assert "3.1" in meta["heading_path"]

    def test_index_metadata_includes_page_number_when_present(self) -> None:
        meta = self._make_chunk().index_metadata()
        assert meta["page_number"] == 4

    def test_index_metadata_omits_page_number_when_absent(self) -> None:
        loc = Location(filename="doc.docx", heading_number="3.1")
        chunk = Chunk(
            id="B::3.1::chunk-0", section_id="B::3.1", doc_id="B",
            chunk_type="prose", text="t", display_text="t", location=loc,
        )
        assert "page_number" not in chunk.index_metadata()

    def test_citation_delegates_to_location_cite(self) -> None:
        chunk = self._make_chunk()
        assert chunk.citation() == chunk.location.cite()
        assert "doc.pdf" in chunk.citation()
        assert "§3.1" in chunk.citation()


# ── ComparisonResult ──────────────────────────────────────────────────────────

class TestComparisonResultSchema:
    def test_serializes_to_exact_brief_schema_keys(self) -> None:
        result = ComparisonResult(
            missing=[MissingEntry(text="t", source_file="f", location="l", explanation="e")],
            diff=[DiffEntry(docA_text="a", docB_text="b", reason="r", sourceA="sa", sourceB="sb")],
            match=[MatchEntry(textA="x", textB="y", source="s")],
        )
        d = result.model_dump()
        assert set(d.keys()) == {"missing", "diff", "match"}

    def test_missing_entry_has_required_keys(self) -> None:
        entry = MissingEntry(text="t", source_file="f.pdf", location="loc", explanation="exp")
        d = entry.model_dump()
        assert {"text", "source_file", "location", "explanation"} == set(d.keys())

    def test_diff_entry_has_required_keys(self) -> None:
        d = DiffEntry(docA_text="a", docB_text="b", reason="r", sourceA="sa", sourceB="sb")
        assert {"docA_text", "docB_text", "reason", "sourceA", "sourceB"} == set(d.model_dump().keys())

    def test_match_entry_has_required_keys(self) -> None:
        d = MatchEntry(textA="a", textB="b", source="s")
        assert {"textA", "textB", "source"} == set(d.model_dump().keys())


# ── ChatAnswer ────────────────────────────────────────────────────────────────

class TestChatAnswer:
    def test_insufficient_context_defaults_to_false(self) -> None:
        answer = ChatAnswer(answer="Some answer.")
        assert answer.insufficient_context is False

    def test_citation_format_with_page(self) -> None:
        c = Citation(filename="doc.pdf", section="3.1", page=5)
        assert c.format() == "doc.pdf · §3.1 · page 5"

    def test_citation_format_without_page(self) -> None:
        c = Citation(filename="doc.docx", section="3.1")
        assert c.format() == "doc.docx · §3.1"

    def test_citation_strips_leading_section_symbol_from_llm_output(self) -> None:
        """LLM sometimes copies § from inline examples into the section field."""
        c = Citation(filename="doc.pdf", section="§3.1", page=4)
        assert c.section == "3.1"
        assert "§§" not in c.format()
        assert c.format() == "doc.pdf · §3.1 · page 4"

    def test_no_citation_format_starts_with_double_section_symbol(self) -> None:
        """Regression: structured citations must never produce §§."""
        cases = ["§3.1", "§§3.1", "§Appendix A", "3.1"]
        for raw in cases:
            c = Citation(filename="f.pdf", section=raw)
            assert not c.format().startswith("§§"), (
                f"Double § in format() for section={raw!r}: {c.format()!r}"
            )
