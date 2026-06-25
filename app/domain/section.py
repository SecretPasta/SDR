from __future__ import annotations

from pydantic import BaseModel


class TableData(BaseModel):
    headers: list[str]
    rows: list[list[str]]

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        header_row = "| " + " | ".join(self.headers) + " |"
        sep_row = "| " + " | ".join("---" for _ in self.headers) + " |"
        data_rows = ["| " + " | ".join(row) + " |" for row in self.rows]
        return "\n".join([header_row, sep_row] + data_rows)


class Location(BaseModel):
    filename: str
    page_number: int | None = None
    heading_number: str | None = None
    heading_path: list[str] = []

    def cite(self) -> str:
        section_ref = self.heading_number or (
            self.heading_path[-1] if self.heading_path else ""
        )
        parts = [self.filename]
        if section_ref:
            parts.append(f"§{section_ref}")
        if self.page_number is not None:
            parts.append(f"page {self.page_number}")
        return " · ".join(parts)


class Section(BaseModel):
    id: str
    location: Location
    heading: str
    body_text: str = ""
    tables: list[TableData] = []
    bullets: list[str] = []

    def for_embedding(self) -> str:
        breadcrumb = " > ".join(self.location.heading_path)
        parts: list[str] = [breadcrumb] if breadcrumb else []
        if self.body_text:
            parts.append(self.body_text)
        if self.bullets:
            parts.extend(self.bullets)
        return "\n".join(parts)


class ParsedDoc(BaseModel):
    doc_id: str
    filename: str
    sections: list[Section]

    def by_heading_number(self) -> dict[str, Section]:
        return {
            s.location.heading_number: s
            for s in self.sections
            if s.location.heading_number is not None
        }
