"""Tests for chat prompt builders — pure functions, no I/O."""
from __future__ import annotations

import pytest

from app.chat.retriever import RetrievedChunk
from app.prompts.chat_synthesis import (
    build_cross_doc_prompt,
    build_single_doc_prompt,
    _cite,
    _doc_label,
)


def _chunk(
    display: str,
    doc_id: str = "A",
    filename: str = "doc.pdf",
    heading_number: str = "3.1",
    page: int | None = 4,
    score: float = 0.9,
) -> RetrievedChunk:
    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "heading_number": heading_number,
        "display_text": display,
    }
    if page is not None:
        meta["page_number"] = page
    return RetrievedChunk(
        id=f"{doc_id}-chunk",
        text=display,
        display_text=display,
        metadata=meta,
        score=score,
    )


class TestBuildSingleDocPrompt:
    def test_prompt_contains_query(self) -> None:
        chunks = [_chunk("Some content")]
        prompt = build_single_doc_prompt("What is the pricing model?", chunks)
        assert "What is the pricing model?" in prompt

    def test_prompt_contains_display_text_of_each_chunk(self) -> None:
        chunks = [_chunk("Alpha content"), _chunk("Beta content")]
        prompt = build_single_doc_prompt("query", chunks)
        assert "Alpha content" in prompt
        assert "Beta content" in prompt

    def test_prompt_contains_citation_in_bracket_format(self) -> None:
        chunks = [_chunk("Content", filename="FDS_V0.pdf", heading_number="3.1", page=5)]
        prompt = build_single_doc_prompt("q", chunks)
        assert "[FDS_V0.pdf · §3.1 · page 5]" in prompt

    def test_prompt_instructs_context_only_no_prior_knowledge(self) -> None:
        prompt = build_single_doc_prompt("q", [_chunk("c")])
        lower = prompt.lower()
        assert "only" in lower or "context" in lower
        assert "prior knowledge" in lower

    def test_prompt_mentions_insufficient_context_flag(self) -> None:
        prompt = build_single_doc_prompt("q", [_chunk("c")])
        assert "insufficient_context" in prompt


class TestBuildCrossDocPrompt:
    def test_prompt_contains_v0_pdf_label(self) -> None:
        chunks_by_doc = {
            "A": [_chunk("V0 content", doc_id="A", filename="FDS_V0.pdf")],
            "B": [_chunk("V5 content", doc_id="B", filename="FDS_V5.docx")],
        }
        prompt = build_cross_doc_prompt("Compare the two.", chunks_by_doc)
        assert "V0 (PDF)" in prompt

    def test_prompt_contains_v5_docx_label(self) -> None:
        chunks_by_doc = {
            "A": [_chunk("V0 content", doc_id="A", filename="FDS_V0.pdf")],
            "B": [_chunk("V5 content", doc_id="B", filename="FDS_V5.docx")],
        }
        prompt = build_cross_doc_prompt("Compare.", chunks_by_doc)
        assert "V5 (DOCX)" in prompt

    def test_prompt_contains_query(self) -> None:
        chunks_by_doc = {"A": [_chunk("c", filename="a.pdf")]}
        prompt = build_cross_doc_prompt("What changed?", chunks_by_doc)
        assert "What changed?" in prompt

    def test_prompt_instructs_attribution_per_source(self) -> None:
        chunks_by_doc = {
            "A": [_chunk("c", filename="a.pdf")],
            "B": [_chunk("d", filename="b.docx")],
        }
        prompt = build_cross_doc_prompt("q", chunks_by_doc)
        lower = prompt.lower()
        assert "attribute" in lower or "source" in lower

    def test_both_doc_texts_appear_in_prompt(self) -> None:
        chunks_by_doc = {
            "A": [_chunk("Alpha section text", filename="a.pdf")],
            "B": [_chunk("Beta section text", filename="b.docx")],
        }
        prompt = build_cross_doc_prompt("q", chunks_by_doc)
        assert "Alpha section text" in prompt
        assert "Beta section text" in prompt


class TestCiteHelper:
    def test_includes_filename_section_and_page(self) -> None:
        meta = {"filename": "doc.pdf", "heading_number": "3.1", "page_number": 5}
        assert _cite(meta) == "doc.pdf · §3.1 · page 5"

    def test_omits_page_when_absent(self) -> None:
        meta = {"filename": "doc.docx", "heading_number": "2.0"}
        assert _cite(meta) == "doc.docx · §2.0"

    def test_falls_back_to_heading_path_when_no_number(self) -> None:
        meta = {"filename": "doc.pdf", "heading_path": ["Introduction"], "page_number": 1}
        result = _cite(meta)
        assert "Introduction" in result

    def test_unknown_filename_when_missing(self) -> None:
        assert _cite({}).startswith("unknown")


class TestDocLabelHelper:
    def test_pdf_extension_returns_v0_label(self) -> None:
        chunks = [_chunk("c", filename="FDS_V0.pdf")]
        assert _doc_label("A", chunks) == "V0 (PDF)"

    def test_docx_extension_returns_v5_label(self) -> None:
        chunks = [_chunk("c", filename="FDS_V5.docx")]
        assert _doc_label("B", chunks) == "V5 (DOCX)"

    def test_empty_chunks_falls_back_to_doc_id(self) -> None:
        assert _doc_label("X", []) == "doc X"
