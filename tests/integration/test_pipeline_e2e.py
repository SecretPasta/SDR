"""End-to-end integration test — requires real API keys and sample files.

Run with: pytest -m integration
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SAMPLE_PDF = Path(__file__).parent.parent.parent / "samples" / "FDS_PriceBook_V0.pdf"
SAMPLE_DOCX = Path(__file__).parent.parent.parent / "samples" / "FDS_PriceBook_V5.docx"

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.skipif(
    not SAMPLE_PDF.exists() or not SAMPLE_DOCX.exists(),
    reason="Sample files not present",
)
async def test_full_pipeline_end_to_end() -> None:
    """Parse both docs, run comparison pipeline, assert non-trivial results."""
    from app.adapters.claude_llm import ClaudeClient
    from app.adapters.gemini_embedder import GeminiEmbedder
    from app.comparison.aligner import HeadingAligner
    from app.comparison.explainer import MissingExplainer
    from app.comparison.judge import PairwiseJudge
    from app.comparison.pipeline import ComparisonPipeline
    from app.comparison.ranker import Top10Ranker
    from app.config import get_settings
    from app.parsing.docx_parser import parse_docx
    from app.parsing.pdf_parser import parse_pdf

    settings = get_settings()

    embedder = GeminiEmbedder(settings.gemini)
    llm = ClaudeClient(settings.anthropic)
    aligner = HeadingAligner(embedder, settings.alignment)
    judge = PairwiseJudge(llm)
    explainer = MissingExplainer(llm)
    ranker = Top10Ranker(llm)

    pipeline = ComparisonPipeline(
        llm=llm,
        embedder=embedder,
        aligner=aligner,
        judge=judge,
        explainer=explainer,
        ranker=ranker,
    )

    doc_a = parse_pdf(SAMPLE_PDF, "A")
    doc_b = parse_docx(SAMPLE_DOCX, "B")

    result, summary = await pipeline.run(doc_a, doc_b)

    # Non-trivial results in at least two categories
    non_empty = sum([
        len(result.match) > 0,
        len(result.diff) > 0,
        len(result.missing) > 0,
    ])
    assert non_empty >= 2, f"Expected changes in ≥2 categories; got {result.model_dump()}"

    # Executive summary capped at 10
    assert len(summary.top_changes) <= 10

    # Totals consistent
    assert summary.total_matches == len(result.match)
    assert summary.total_diffs == len(result.diff)
    assert summary.total_missing == len(result.missing)

    # Save output for manual inspection
    output_dir = Path(__file__).parent / "_outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "comparison_result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_dir / "executive_summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
