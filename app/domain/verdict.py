from typing import Literal

from pydantic import BaseModel

from app.domain.comparison import MissingEntry


class PairwiseVerdict(BaseModel):
    section_id_a: str
    section_id_b: str
    verdict: Literal["MATCH", "DIFF"]
    reason: str | None = None  # populated for DIFF
    doc_a_text: str = ""
    doc_b_text: str = ""
    source_a: str = ""  # citation string for doc A
    source_b: str = ""  # citation string for doc B


class MissingExplanationBatch(BaseModel):
    entries: list[MissingEntry]