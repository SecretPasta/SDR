"""Comparison endpoints: POST /compare, GET /summary."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import CompareRequest, CompareResponse, SummaryResponse
from app.deps import ComparisonPipelineDep
from app.domain.comparison import ComparisonResult
from app.domain.summary import ExecutiveSummary
from app.parsing.docx_parser import parse_docx
from app.parsing.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)
router = APIRouter(tags=["compare"])

_OUTPUTS = Path(__file__).resolve().parent.parent.parent / "outputs"

# Module-level cache — populated by POST /compare, read by GET /summary.
_cached_result: ComparisonResult | None = None
_cached_summary: ExecutiveSummary | None = None


def _write_outputs(result: ComparisonResult, summary: ExecutiveSummary) -> None:
    _OUTPUTS.mkdir(exist_ok=True)
    (_OUTPUTS / "comparison.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (_OUTPUTS / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote outputs/comparison.json and outputs/summary.json")


@router.post("/compare", response_model=CompareResponse)
async def compare(body: CompareRequest, pipeline: ComparisonPipelineDep) -> CompareResponse:
    global _cached_result, _cached_summary

    pdf_path = Path(body.pdf_path)
    docx_path = Path(body.docx_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=422, detail=f"PDF not found: {body.pdf_path}")
    if not docx_path.exists():
        raise HTTPException(status_code=422, detail=f"DOCX not found: {body.docx_path}")

    logger.info("Parsing documents: %s, %s", pdf_path.name, docx_path.name)
    doc_a = parse_pdf(pdf_path, doc_id="A")
    doc_b = parse_docx(docx_path, doc_id="B")

    logger.info("Running comparison pipeline")
    result, summary = await pipeline.run(doc_a, doc_b)

    _cached_result = result
    _cached_summary = summary

    _write_outputs(result, summary)

    return CompareResponse(result=result, summary=summary)


@router.get("/summary", response_model=SummaryResponse)
async def get_summary() -> SummaryResponse:
    if _cached_summary is None:
        raise HTTPException(
            status_code=404,
            detail="No comparison has been run yet. Call POST /compare first.",
        )
    return SummaryResponse(summary=_cached_summary)
