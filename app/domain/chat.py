from pydantic import BaseModel


class Citation(BaseModel):
    filename: str
    section: str
    page: int | None = None

    def format(self) -> str:
        parts = [self.filename, f"§{self.section}"]
        if self.page is not None:
            parts.append(f"page {self.page}")
        return " · ".join(parts)


class ChatAnswer(BaseModel):
    answer: str
    citations: list[Citation] = []
    insufficient_context: bool = False