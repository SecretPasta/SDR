"""Tests for PairwiseJudge — LLM-powered MATCH/DIFF verdict."""
from __future__ import annotations

import pytest

from app.comparison.judge import PairwiseJudge
from app.domain.section import Location, Section
from app.domain.verdict import PairwiseVerdict
from tests.fakes.fake_llm import FakeLLM


def _sec(doc_id: str, num: str, body: str) -> Section:
    return Section(
        id=f"{doc_id}::{num}",
        location=Location(filename=f"{doc_id}.pdf", heading_number=num,
                          heading_path=[num, "Section"]),
        heading="Section",
        body_text=body,
    )


class TestPairwiseJudge:
    async def test_judge_returns_pairwise_verdict(self) -> None:
        llm = FakeLLM([{"verdict": "MATCH", "reason": "identical"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "1.0", "Same content")
        sb = _sec("B", "1.0", "Same content")

        result = await judge.judge(sa, sb)

        assert isinstance(result, PairwiseVerdict)
        assert result.verdict == "MATCH"

    async def test_judge_makes_exactly_one_llm_call(self) -> None:
        llm = FakeLLM([{"verdict": "DIFF", "reason": "content differs"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "1.0", "Old content")
        sb = _sec("B", "1.0", "New content")

        await judge.judge(sa, sb)

        assert len(llm.calls) == 1

    async def test_judge_prompt_contains_both_section_texts(self) -> None:
        llm = FakeLLM([{"verdict": "MATCH", "reason": "same"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "1.0", "Alpha content text")
        sb = _sec("B", "1.0", "Beta content text")

        await judge.judge(sa, sb)

        prompt_text = llm.calls[0][0][0]["content"]
        assert "Alpha content text" in prompt_text
        assert "Beta content text" in prompt_text

    async def test_judge_verdict_carries_section_ids(self) -> None:
        llm = FakeLLM([{"verdict": "DIFF", "reason": "changed"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "2.0", "Old")
        sb = _sec("B", "2.0", "New")

        result = await judge.judge(sa, sb)

        assert result.section_id_a == sa.id
        assert result.section_id_b == sb.id

    async def test_diff_verdict_includes_reason(self) -> None:
        llm = FakeLLM([{"verdict": "DIFF", "reason": "thresholds changed"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "3.0", "Threshold is 5%")
        sb = _sec("B", "3.0", "Threshold is 10%")

        result = await judge.judge(sa, sb)

        assert result.verdict == "DIFF"
        assert result.reason == "thresholds changed"

    async def test_match_verdict_has_no_reason(self) -> None:
        llm = FakeLLM([{"verdict": "MATCH", "reason": "identical"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "1.0", "Same")
        sb = _sec("B", "1.0", "Same")

        result = await judge.judge(sa, sb)

        assert result.verdict == "MATCH"
        assert result.reason is None

    async def test_llm_call_passes_response_schema(self) -> None:
        llm = FakeLLM([{"verdict": "MATCH", "reason": "ok"}])
        judge = PairwiseJudge(llm)
        sa = _sec("A", "1.0", "x")
        sb = _sec("B", "1.0", "x")

        await judge.judge(sa, sb)

        _, schema = llm.calls[0]
        assert schema is not None
        assert schema.__name__ == "_JudgeOutput"
