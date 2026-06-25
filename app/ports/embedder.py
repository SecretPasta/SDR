from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderClient(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts for indexing (RETRIEVAL_DOCUMENT task)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval (RETRIEVAL_QUERY task)."""
        ...

    async def embed_for_similarity(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for heading alignment scoring (SEMANTIC_SIMILARITY task)."""
        ...