"""Run the full comparison pipeline and write outputs/comparison.json + summary.json."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from app.domain.comparison import ComparisonResult
from app.domain.summary import ExecutiveSummary

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS = _ROOT / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    from app.adapters.claude_llm import ClaudeClient
    from app.ports.llm import LLMClient
    from app.adapters.gemini_embedder import GeminiEmbedder
    from app.comparison.aligner import HeadingAligner
    from app.comparison.explainer import MissingExplainer
    from app.comparison.judge import PairwiseJudge
    from app.comparison.pipeline import ComparisonPipeline
    from app.comparison.ranker import Top10Ranker
    from app.config import get_settings
    from scripts.index_docs import load_cached_docs

    try:
        doc_a, doc_b = load_cached_docs()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    settings = get_settings()
    llm: LLMClient = ClaudeClient(settings.anthropic)  # type: ignore[assignment]
    embedder = GeminiEmbedder(settings.gemini)
    aligner  = HeadingAligner(embedder, settings.alignment)
    judge    = PairwiseJudge(llm)
    explainer = MissingExplainer(llm)
    ranker   = Top10Ranker(llm)

    pipeline = ComparisonPipeline(
        llm=llm,
        embedder=embedder,
        aligner=aligner,
        judge=judge,
        explainer=explainer,
        ranker=ranker,
    )

    logger.info("Running comparison pipeline …")
    result, summary = await pipeline.run(doc_a, doc_b)

    _write(result, summary)

    logger.info(
        "Done — %d diffs, %d missing, %d matches, %d ranked changes",
        len(result.diff), len(result.missing), len(result.match), len(summary.top_changes),
    )


def _write(result: ComparisonResult, summary: ExecutiveSummary) -> None:
    _OUTPUTS.mkdir(exist_ok=True)
    comparison_path = _OUTPUTS / "comparison.json"
    summary_path    = _OUTPUTS / "summary.json"
    comparison_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote %s", comparison_path)
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    asyncio.run(main())