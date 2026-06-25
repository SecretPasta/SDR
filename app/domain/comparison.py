from pydantic import BaseModel


class MissingEntry(BaseModel):
    text: str
    source_file: str
    location: str
    explanation: str


class DiffEntry(BaseModel):
    docA_text: str
    docB_text: str
    reason: str
    sourceA: str
    sourceB: str


class MatchEntry(BaseModel):
    textA: str
    textB: str
    source: str


class ComparisonResult(BaseModel):
    missing: list[MissingEntry] = []
    diff: list[DiffEntry] = []
    match: list[MatchEntry] = []