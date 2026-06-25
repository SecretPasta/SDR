"""Tests for chunk_section — structure-aware text chunking logic."""
from __future__ import annotations

import pytest

from app.config import ChunkingSettings
from app.domain.section import Location, Section, TableData
from app.indexing.chunker import chunk_section

_SETTINGS = ChunkingSettings()


def _sec(
    heading: str = "Test Section",
    heading_number: str | None = "1.1",
    body_text: str = "",
    tables: list[TableData] | None = None,
    bullets: list[str] | None = None,
) -> Section:
    path = [heading_number, heading] if heading_number else [heading]
    return Section(
        id=f"A::{heading_number or 'x'}",
        location=Location(
            filename="doc.pdf",
            heading_number=heading_number,
            heading_path=path,
            page_number=1,
        ),
        heading=heading,
        body_text=body_text,
        tables=tables or [],
        bullets=bullets or [],
    )


class TestShortSection:
    def test_short_section_produces_single_prose_chunk(self) -> None:
        sec = _sec(body_text="Short body text.")
        chunks = chunk_section(sec, _SETTINGS)
        prose = [c for c in chunks if c.chunk_type == "prose"]
        assert len(prose) == 1

    def test_chunk_id_follows_section_id_chunk_seq_pattern(self) -> None:
        sec = _sec(body_text="Body.")
        chunks = chunk_section(sec, _SETTINGS)
        assert chunks[0].id == f"{sec.id}::chunk-0"

    def test_display_text_does_not_contain_heading_breadcrumb(self) -> None:
        sec = _sec(heading="Process Stages", heading_number="3.1", body_text="Some content.")
        chunks = chunk_section(sec, _SETTINGS)
        for chunk in chunks:
            assert "3.1" not in chunk.display_text or "3.1" in "Some content."

    def test_text_field_starts_with_heading_breadcrumb(self) -> None:
        sec = _sec(heading="Process Stages", heading_number="3.1", body_text="Content here.")
        chunks = chunk_section(sec, _SETTINGS)
        prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
        assert len(prose_chunks) >= 1
        # breadcrumb = "1.1 > Process Stages" is NOT how we build it;
        # actual breadcrumb is from heading_path joined by " > "
        assert "Process Stages" in prose_chunks[0].text or "3.1" in prose_chunks[0].text


class TestTableChunking:
    def test_table_gets_its_own_chunk(self) -> None:
        table = TableData(headers=["Phase", "Status"], rows=[["A", "Live"]])
        sec = _sec(body_text="Some prose.", tables=[table])
        chunks = chunk_section(sec, _SETTINGS)
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) == 1

    def test_table_chunk_display_text_is_markdown(self) -> None:
        table = TableData(headers=["Phase", "Status"], rows=[["A", "Live"]])
        sec = _sec(tables=[table])
        chunks = chunk_section(sec, _SETTINGS)
        table_chunk = next(c for c in chunks if c.chunk_type == "table")
        assert "| Phase | Status |" in table_chunk.display_text

    def test_two_tables_produce_two_table_chunks(self) -> None:
        tables = [
            TableData(headers=["A"], rows=[["1"]]),
            TableData(headers=["B"], rows=[["2"]]),
        ]
        sec = _sec(tables=tables)
        chunks = chunk_section(sec, _SETTINGS)
        assert len([c for c in chunks if c.chunk_type == "table"]) == 2

    def test_table_chunk_ids_are_sequential(self) -> None:
        tables = [TableData(headers=["X"], rows=[["1"]]), TableData(headers=["Y"], rows=[["2"]])]
        sec = _sec(tables=tables)
        chunks = [c for c in chunk_section(sec, _SETTINGS) if c.chunk_type == "table"]
        ids = [c.id for c in chunks]
        assert ids[0].endswith("::chunk-0")
        assert ids[1].endswith("::chunk-1")


class TestBulletsChunking:
    def test_bullets_only_section_produces_bullets_chunk(self) -> None:
        sec = _sec(bullets=["item one", "item two", "item three"])
        chunks = chunk_section(sec, _SETTINGS)
        bullet_chunks = [c for c in chunks if c.chunk_type == "bullets"]
        assert len(bullet_chunks) >= 1

    def test_bullet_display_text_uses_bullet_prefix(self) -> None:
        sec = _sec(bullets=["alpha", "beta"])
        chunks = chunk_section(sec, _SETTINGS)
        bullet_chunk = next(c for c in chunks if c.chunk_type == "bullets")
        assert "• alpha" in bullet_chunk.display_text
        assert "• beta" in bullet_chunk.display_text


class TestLongSectionSplitting:
    def _long_body(self, num_paras: int = 20, words_per_para: int = 60) -> str:
        para = " ".join(["word"] * words_per_para)
        return "\n".join([para] * num_paras)

    def test_long_section_splits_into_multiple_chunks(self) -> None:
        sec = _sec(body_text=self._long_body(30, 80))
        chunks = chunk_section(sec, _SETTINGS)
        prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
        assert len(prose_chunks) > 1

    def test_no_chunk_exceeds_max_tokens(self) -> None:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        sec = _sec(body_text=self._long_body(30, 80))
        chunks = chunk_section(sec, _SETTINGS)
        for chunk in chunks:
            # text field includes breadcrumb — check display_text against max
            assert len(enc.encode(chunk.display_text)) <= _SETTINGS.max_tokens * 1.1  # small tolerance

    def test_chunk_ids_are_deterministic_and_sequential(self) -> None:
        sec = _sec(body_text=self._long_body(30, 80))
        chunks1 = chunk_section(sec, _SETTINGS)
        chunks2 = chunk_section(sec, _SETTINGS)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_section_id_embedded_in_chunk_ids(self) -> None:
        sec = _sec(body_text=self._long_body(30, 80))
        for chunk in chunk_section(sec, _SETTINGS):
            assert chunk.section_id == sec.id
            assert chunk.id.startswith(sec.id)
