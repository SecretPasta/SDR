"""Tests for Top10Ranker — LLM-powered executive summary / change ranking."""
from __future__ import annotations

import pytest

from app.comparison.ranker import Top10Ranker
from app.domain.comparison import ComparisonResult, DiffEntry, MatchEntry, MissingEntry
from app.domain.summary import ExecutiveSummary
from tests.fakes.fake_llm import FakeLLM


def _diff(i: int) -> DiffEntry:
    return DiffEntry(
        docA_text=f"Old text {i}",
        docB_text=f"New text {i}",
        reason=f"reason {i}",
        sourceA=f"A.pdf · §{i}.0",
        sourceB=f"B.docx · §{i}.0",
    )


def _missing(i: int) -> MissingEntry:
    return MissingEntry(
        text=f"Missing section {i}",
        source_file="A.pdf",
        location=f"A.pdf · §{i}.0",
        explanation=f"Removed in V5 ({i})",
    )


def _make_result(n_diff: int = 0, n_missing: int = 0, n_match: int = 0) -> ComparisonResult:
    return ComparisonResult(
        diff=[_diff(i) for i in range(n_diff)],
        missing=[_missing(i) for i in range(n_missing)],
        match=[MatchEntry(textA="x", textB="x", source=f"s{i}") for i in range(n_match)],
    )


def _top10_response(entry_ids: list[str]) -> dict:
    return {
        "changes": [
            {
                "rank": i + 1,
                "change_type": "scope change",
                "summary": f"Change {i}",
                "why_it_matters": "important",
                "citations": [],
                "source_entry_id": eid,
            }
            for i, eid in enumerate(entry_ids)
        ]
    }


class TestTop10RankerEmptyInput:
    async def test_no_diff_or_missing_returns_empty_summary_without_llm_call(self) -> None:
        llm = FakeLLM([])
        ranker = Top10Ranker(llm)
        result = _make_result(n_match=5)

        summary = await ranker.rank(result)

        assert isinstance(summary, ExecutiveSummary)
        assert summary.top_changes == []
        assert len(llm.calls) == 0

    async def test_totals_reflect_input_counts(self) -> None:
        llm = FakeLLM([])
        ranker = Top10Ranker(llm)
        result = _make_result(n_match=3)

        summary = await ranker.rank(result)

        assert summary.total_matches == 3
        assert summary.total_diffs == 0
        assert summary.total_missing == 0


class TestTop10RankerWithContent:
    async def test_returns_at_most_10_changes(self) -> None:
        result = _make_result(n_diff=10, n_missing=5)
        # Return 15 entries; ranker should cap at 10
        ids = [f"diff::{i}" for i in range(10)] + [f"missing::{i}" for i in range(5)]
        llm = FakeLLM([_top10_response(ids)])
        ranker = Top10Ranker(llm)

        summary = await ranker.rank(result)

        assert len(summary.top_changes) <= 10

    async def test_makes_exactly_one_llm_call(self) -> None:
        result = _make_result(n_diff=3)
        llm = FakeLLM([_top10_response(["diff::0", "diff::1", "diff::2"])])
        ranker = Top10Ranker(llm)

        await ranker.rank(result)

        assert len(llm.calls) == 1

    async def test_diff_source_id_produces_diff_verdict(self) -> None:
        result = _make_result(n_diff=2)
        llm = FakeLLM([_top10_response(["diff::0", "diff::1"])])
        ranker = Top10Ranker(llm)

        summary = await ranker.rank(result)

        assert all(c.verdict == "DIFF" for c in summary.top_changes)

    async def test_missing_source_id_produces_missing_verdict(self) -> None:
        result = _make_result(n_missing=2)
        llm = FakeLLM([_top10_response(["missing::0", "missing::1"])])
        ranker = Top10Ranker(llm)

        summary = await ranker.rank(result)

        assert all(c.verdict == "MISSING" for c in summary.top_changes)

    async def test_totals_match_input_regardless_of_top10(self) -> None:
        result = _make_result(n_diff=8, n_missing=3, n_match=5)
        ids = [f"diff::{i}" for i in range(8)] + [f"missing::{i}" for i in range(2)]
        llm = FakeLLM([_top10_response(ids[:10])])
        ranker = Top10Ranker(llm)

        summary = await ranker.rank(result)

        assert summary.total_diffs == 8
        assert summary.total_missing == 3
        assert summary.total_matches == 5

    async def test_llm_receives_response_schema(self) -> None:
        result = _make_result(n_diff=1)
        llm = FakeLLM([_top10_response(["diff::0"])])
        ranker = Top10Ranker(llm)

        await ranker.rank(result)

        _, schema = llm.calls[0]
        assert schema is not None
        assert schema.__name__ == "_Top10Output"

    async def test_changes_sorted_by_rank(self) -> None:
        result = _make_result(n_diff=3)
        # Return out of order
        response = {
            "changes": [
                {"rank": 3, "change_type": "", "summary": "C", "why_it_matters": "",
                 "citations": [], "source_entry_id": "diff::2"},
                {"rank": 1, "change_type": "", "summary": "A", "why_it_matters": "",
                 "citations": [], "source_entry_id": "diff::0"},
                {"rank": 2, "change_type": "", "summary": "B", "why_it_matters": "",
                 "citations": [], "source_entry_id": "diff::1"},
            ]
        }
        llm = FakeLLM([response])
        ranker = Top10Ranker(llm)

        summary = await ranker.rank(result)

        ranks = [c.rank for c in summary.top_changes]
        assert ranks == sorted(ranks)
