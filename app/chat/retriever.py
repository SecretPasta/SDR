from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from app.ports.embedder import EmbedderClient
from app.ports.vector_store import VectorStore


class RetrievedChunk(BaseModel):
    id: str
    text: str
    display_text: str
    metadata: dict[str, Any]
    score: float


class Retriever:
    def __init__(self, embedder: EmbedderClient, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve_single(
        self,
        query: str,
        doc_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        vector = await self._embedder.embed_query(query)
        results = await self._store.query(
            vector,
            top_k=top_k,
            metadata_filter={"doc_id": doc_id},
        )
        return [
            RetrievedChunk(
                id=r["id"],
                text=r["metadata"].get("display_text", ""),
                display_text=r["metadata"].get("display_text", ""),
                metadata=r["metadata"],
                score=r["score"],
            )
            for r in results
        ]

    async def retrieve_dual(
        self,
        query: str,
        doc_ids: tuple[str, str],
        top_k_per_doc: int,
    ) -> dict[str, list[RetrievedChunk]]:
        vector = await self._embedder.embed_query(query)

        async def _query(doc_id: str) -> tuple[str, list[RetrievedChunk]]:
            results = await self._store.query(
                vector,
                top_k=top_k_per_doc,
                metadata_filter={"doc_id": doc_id},
            )
            chunks = [
                RetrievedChunk(
                    id=r["id"],
                    text=r["metadata"].get("display_text", ""),
                    display_text=r["metadata"].get("display_text", ""),
                    metadata=r["metadata"],
                    score=r["score"],
                )
                for r in results
            ]
            return doc_id, chunks

        pairs = await asyncio.gather(*[_query(did) for did in doc_ids])
        return {doc_id: chunks for doc_id, chunks in pairs}

    async def retrieve_unfiltered(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Single query across all documents — no doc_id filter."""
        vector = await self._embedder.embed_query(query)
        results = await self._store.query(vector, top_k=top_k)
        return [
            RetrievedChunk(
                id=r["id"],
                text=r["metadata"].get("display_text", ""),
                display_text=r["metadata"].get("display_text", ""),
                metadata=r["metadata"],
                score=r["score"],
            )
            for r in results
        ]
