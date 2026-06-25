"""API request/response schemas — kept separate from domain models."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.comparison import ComparisonResult
from app.domain.summary import ExecutiveSummary


# ── /compare ──────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    pdf_path: str = Field(description="Path to the V0 PDF file")
    docx_path: str = Field(description="Path to the V5 DOCX file")


class CompareResponse(BaseModel):
    result: ComparisonResult
    summary: ExecutiveSummary


# ── /summary ──────────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    summary: ExecutiveSummary


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language question")
    doc_id: str | None = Field(
        default=None,
        description="Target doc_id for single-doc chat (required for /chat/single)",
    )


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = Field(
        default_factory=list,
        description="Formatted citation strings (filename · §section · page N)",
    )
    insufficient_context: bool = False
