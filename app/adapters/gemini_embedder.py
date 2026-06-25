from __future__ import annotations

import logging

from google import genai
from google.genai import types

from app.config import GeminiSettings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # gemini-embedding-001 per-call hard limit


class GeminiEmbedder:
    def __init__(self, settings: GeminiSettings) -> None:
        self._client = genai.Client(api_key=settings.api_key.get_secret_value())
        self._model = settings.embed_model
        self._dimensions = settings.embed_dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        results = await self._embed([text], task_type="RETRIEVAL_QUERY")
        return results[0]

    async def embed_for_similarity(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, task_type="SEMANTIC_SIMILARITY")

    async def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not texts:
            return []
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions,
        )
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=batch,
                config=config,
            )
            embeddings = response.embeddings or []
            for e in embeddings:
                if e.values is None:
                    raise ValueError("Gemini returned an embedding with no values")
                results.append(e.values)
        return results