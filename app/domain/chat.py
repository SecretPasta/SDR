from pydantic import BaseModel, field_validator


class Citation(BaseModel):
    filename: str
    section: str
    page: int | None = None

    @field_validator("section")
    @classmethod
    def strip_section_symbol(cls, v: str) -> str:
        """LLMs sometimes copy the § from inline citation examples into this field.
        Strip it so format() doesn't produce §§3.1."""
        return v.lstrip("§")

    def format(self) -> str:
        parts = [self.filename, f"§{self.section}"]
        if self.page is not None:
            parts.append(f"page {self.page}")
        return " · ".join(parts)


class ChatAnswer(BaseModel):
    answer: str
    citations: list[Citation] = []
    insufficient_context: bool = False