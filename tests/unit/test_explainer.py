"""Tests for MissingExplainer — batched LLM explanation of unmatched sections."""
from __future__ import annotations

import pytest

from app.comparison.explainer import MissingExplainer
from app.domain.section import Location, Section
from app.domain.verdict import MissingExplanationBatch
from tests.fakes.fake_llm import FakeLLM


def _sec(doc_id: str, num: str, heading: str, body: str = "body", filename: str | None = None) -> Section:
    fname = filename or f"{doc_id}.pdf"
    return Section(
        id=f"{doc_id}::{num}",
        location=Location(filename=fname, heading_number=num,
                          heading_path=[num, heading]),
        heading=heading,
        body_text=body,
    )


class TestMissingExplainer:
    async def test_empty_input_returns_empty_batch_without_llm_call(self) -> None:
        llm = FakeLLM([])
        explainer = MissingExplainer(llm)

        result = await explainer.explain([], [])

        assert isinstance(result, MissingExplanationBatch)
        assert result.entries == []
        assert len(llm.calls) == 0

    async def test_makes_exactly_one_llm_call_regardless_of_section_count(self) -> None:
        sections_a = [_sec("A", f"{i}.0", f"Section {i}") for i in range(1, 4)]
        sections_b = [_sec("B", f"{i}.0", f"Section {i}") for i in range(10, 13)]
        llm = FakeLLM([{"explanations": []}])
        explainer = MissingExplainer(llm)

        await explainer.explain(sections_a, sections_b)

        assert len(llm.calls) == 1

    async def test_output_has_entry_for_each_input_section(self) -> None:
        sa = _sec("A", "3.0", "Legacy Migration")
        sb = _sec("B", "5.0", "Integration Checks")

        llm = FakeLLM([{
            "explanations": [
                {"section_id": sa.id, "explanation": "Removed in V5."},
                {"section_id": sb.id, "explanation": "New in V5."},
            ]
        }])
        explainer = MissingExplainer(llm)

        result = await explainer.explain([sa], [sb])

        assert len(result.entries) == 2

    async def test_prompt_contains_all_section_headings(self) -> None:
        sa = _sec("A", "3.0", "Legacy Migration")
        sb = _sec("B", "5.0", "Integration Alpha")
        llm = FakeLLM([{"explanations": []}])
        explainer = MissingExplainer(llm)

        await explainer.explain([sa], [sb])

        prompt = llm.calls[0][0][0]["content"]
        assert "Legacy Migration" in prompt
        assert "Integration Alpha" in prompt

    async def test_unknown_section_id_from_llm_gets_default_explanation(self) -> None:
        sa = _sec("A", "3.0", "Missing Section")
        llm = FakeLLM([{"explanations": []}])  # LLM returns no explanations
        explainer = MissingExplainer(llm)

        result = await explainer.explain([sa], [])

        assert len(result.entries) == 1
        assert result.entries[0].explanation != ""

    async def test_llm_call_passes_response_schema(self) -> None:
        sa = _sec("A", "1.0", "Intro")
        llm = FakeLLM([{"explanations": []}])
        explainer = MissingExplainer(llm)

        await explainer.explain([sa], [])

        _, schema = llm.calls[0]
        assert schema is not None
        assert schema.__name__ == "_ExplainerOutput"

    async def test_result_entries_have_correct_source_file(self) -> None:
        sa = _sec("A", "3.0", "V0 Only", filename="FDS_V0.pdf")
        llm = FakeLLM([{"explanations": []}])
        explainer = MissingExplainer(llm)

        result = await explainer.explain([sa], [])

        assert result.entries[0].source_file == "FDS_V0.pdf"
