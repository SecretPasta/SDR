"""Tests for ComparisonPipeline — full LangGraph orchestration with all-fakes wiring."""
from __future__ import annotations

from typing import Any

import pytest

from app.comparison.aligner import HeadingAligner
from app.comparison.explainer import MissingExplainer
from app.comparison.judge import PairwiseJudge
from app.comparison.pipeline import ComparisonPipeline
from app.comparison.ranker import Top10Ranker
from app.config import AlignmentSettings
from app.domain.comparison import ComparisonResult
from app.domain.summary import ExecutiveSummary
from tests.fakes.fake_embedder import FakeEmbedder
from tests.fakes.fake_llm import FakeLLM
from tests.fixtures.parsed_docs import make_parsed_doc_a, make_parsed_doc_b


def _llm_handler(messages: list[dict], schema: Any) -> dict:
    """Dispatch correct minimal response based on which schema is requested."""
    if schema is None:
        return {"content": "ok"}
    name = schema.__name__
    if name == "_JudgeOutput":
        return {"verdict": "DIFF", "reason": "content differs between versions"}
    if name == "_ExplainerOutput":
        return {"explanations": []}
    if name == "_Top10Output":
        return {
            "changes": [
                {
                    "rank": 1,
                    "change_type": "content change",
                    "summary": "Price calculation logic updated",
                    "why_it_matters": "Affects billing",
                    "citations": [],
                    "source_entry_id": "diff::0",
                }
            ]
        }
    return {}


def _build_pipeline(fake_llm: FakeLLM) -> ComparisonPipeline:
    fake_embedder = FakeEmbedder()
    aligner = HeadingAligner(fake_embedder, AlignmentSettings())
    judge = PairwiseJudge(fake_llm)
    explainer = MissingExplainer(fake_llm)
    ranker = Top10Ranker(fake_llm)
    return ComparisonPipeline(
        llm=fake_llm,
        embedder=fake_embedder,
        aligner=aligner,
        judge=judge,
        explainer=explainer,
        ranker=ranker,
    )


class TestPipelineEndToEnd:
    async def test_pipeline_produces_comparison_result_and_executive_summary(self) -> None:
        llm = FakeLLM(handler=_llm_handler)
        pipeline = _build_pipeline(llm)
        doc_a = make_parsed_doc_a()
        doc_b = make_parsed_doc_b()

        result, summary = await pipeline.run(doc_a, doc_b)

        assert isinstance(result, ComparisonResult)
        assert isinstance(summary, ExecutiveSummary)

    async def test_pipeline_result_has_entries_in_at_least_two_categories(self) -> None:
        llm = FakeLLM(handler=_llm_handler)
        pipeline = _build_pipeline(llm)
        doc_a = make_parsed_doc_a()
        doc_b = make_parsed_doc_b()

        result, _ = await pipeline.run(doc_a, doc_b)

        non_empty = sum([
            len(result.match) > 0,
            len(result.diff) > 0,
            len(result.missing) > 0,
        ])
        assert non_empty >= 2

    async def test_pipeline_missing_sections_are_detected(self) -> None:
        """doc_a has '4.3 Legacy Migration' with no counterpart in doc_b."""
        llm = FakeLLM(handler=_llm_handler)
        pipeline = _build_pipeline(llm)
        doc_a = make_parsed_doc_a()
        doc_b = make_parsed_doc_b()

        result, _ = await pipeline.run(doc_a, doc_b)

        assert len(result.missing) >= 1

    async def test_pipeline_llm_call_count_matches_pairs_plus_explainer_plus_ranker(self) -> None:
        """
        5 aligned pairs → 5 judge calls.
        2 unmatched (1 in A, 1 in B) → 1 explainer call.
        diffs present → 1 ranker call.
        Total ≥ 7. (Exact count depends on alignment outcome.)
        """
        llm = FakeLLM(handler=_llm_handler)
        pipeline = _build_pipeline(llm)
        doc_a = make_parsed_doc_a()
        doc_b = make_parsed_doc_b()

        result, _ = await pipeline.run(doc_a, doc_b)

        # At minimum: one judge call per aligned pair + explainer (if unmatched) + ranker (if diffs)
        n_aligned = len(result.match) + len(result.diff)
        has_unmatched = len(result.missing) > 0
        has_diffs = len(result.diff) > 0

        min_expected = n_aligned + (1 if has_unmatched else 0) + (1 if has_diffs else 0)
        assert len(llm.calls) >= min_expected

    async def test_executive_summary_totals_are_consistent_with_result(self) -> None:
        llm = FakeLLM(handler=_llm_handler)
        pipeline = _build_pipeline(llm)
        doc_a = make_parsed_doc_a()
        doc_b = make_parsed_doc_b()

        result, summary = await pipeline.run(doc_a, doc_b)

        assert summary.total_matches == len(result.match)
        assert summary.total_diffs == len(result.diff)
        assert summary.total_missing == len(result.missing)

    async def test_pipeline_is_deterministic_on_identical_input(self) -> None:
        """Same docs → same ComparisonResult structure on two runs."""
        doc_a = make_parsed_doc_a()
        doc_b = make_parsed_doc_b()

        llm1 = FakeLLM(handler=_llm_handler)
        result1, _ = await _build_pipeline(llm1).run(doc_a, doc_b)

        llm2 = FakeLLM(handler=_llm_handler)
        result2, _ = await _build_pipeline(llm2).run(doc_a, doc_b)

        assert len(result1.match) == len(result2.match)
        assert len(result1.diff) == len(result2.diff)
        assert len(result1.missing) == len(result2.missing)
