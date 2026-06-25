from typing import Literal

from pydantic import BaseModel


class ImportantChange(BaseModel):
    rank: int
    verdict: Literal["DIFF", "MISSING"]
    summary: str
    section_id_a: str | None = None
    section_id_b: str | None = None


class ExecutiveSummary(BaseModel):
    top_changes: list[ImportantChange]
    total_matches: int
    total_diffs: int
    total_missing: int