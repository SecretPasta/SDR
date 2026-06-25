from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.domain.comparison import ComparisonResult
from app.domain.summary import ExecutiveSummary, ImportantChange
from app.ports.llm import LLMClient
from app.prompts.top10_ranker import build_top10_prompt


class _RankedEntry(BaseModel):
    """Flat schema filled by the LLM — no unions, Gemini compatible."""
    rank: int
    change_type: str
    summary: str
    why_it_matters: str
    citations: list[str]
    source_entry_id: str  # e.g. "diff::0", "missing::3"


class _Top10Output(BaseModel):
    changes: list[_RankedEntry]


class Top10Ranker:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rank(self, result: ComparisonResult) -> ExecutiveSummary:
        if not result.diff and not result.missing:
            return ExecutiveSummary(
                top_changes=[],
                total_matches=len(result.match),
                total_diffs=0,
                total_missing=0,
            )

        prompt = build_top10_prompt(result)
        raw = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            response_schema=_Top10Output,
        )
        output = _Top10Output.model_validate(raw)

        # Build the same stable ID → entry maps the prompt used for back-references
        diff_map    = {f"diff::{i}":    e for i, e in enumerate(result.diff)}
        missing_map = {f"missing::{i}": e for i, e in enumerate(result.missing)}

        top_changes: list[ImportantChange] = []
        for entry in sorted(output.changes, key=lambda e: e.rank)[:10]:
            verdict: Literal["DIFF", "MISSING"] = (
                "DIFF" if entry.source_entry_id.startswith("diff::") else "MISSING"
            )

            # Resolve source citations from the original entries if the LLM omitted them
            if not entry.citations:
                if entry.source_entry_id in diff_map:
                    de = diff_map[entry.source_entry_id]
                    entry.citations = [de.sourceA, de.sourceB]
                elif entry.source_entry_id in missing_map:
                    me = missing_map[entry.source_entry_id]
                    entry.citations = [me.location]

            top_changes.append(ImportantChange(
                rank=entry.rank,
                verdict=verdict,
                change_type=entry.change_type,
                summary=entry.summary,
                why_it_matters=entry.why_it_matters,
                citations=entry.citations,
            ))

        return ExecutiveSummary(
            top_changes=top_changes,
            total_matches=len(result.match),
            total_diffs=len(result.diff),
            total_missing=len(result.missing),
        )