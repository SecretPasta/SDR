from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.domain.section import Location

ChunkType = Literal["prose", "table", "bullets"]


class Chunk(BaseModel):
    id: str
    section_id: str
    doc_id: str
    chunk_type: ChunkType
    text: str          # heading breadcrumb prepended — used for embedding
    display_text: str  # clean prose, no breadcrumb — stored in index metadata
    location: Location

    def index_metadata(self) -> dict[str, str | int | list[str]]:
        meta: dict[str, str | int | list[str]] = {
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "chunk_type": self.chunk_type,
            "display_text": self.display_text,
            "filename": self.location.filename,
            "heading_path": self.location.heading_path,
        }
        if self.location.heading_number is not None:
            meta["heading_number"] = self.location.heading_number
        if self.location.page_number is not None:
            meta["page_number"] = self.location.page_number
        return meta

    def citation(self) -> str:
        return self.location.cite()