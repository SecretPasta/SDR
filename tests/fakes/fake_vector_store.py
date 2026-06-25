"""InMemoryVectorStore — VectorStore implementation for unit tests."""
from __future__ import annotations

import math
from typing import Any

from app.ports.vector_store import QueryResult, VectorRecord


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Implements VectorStore protocol with brute-force cosine similarity.

    Supports optional metadata filter via exact-match on filter keys.
    """

    def __init__(self) -> None:
        self._store: dict[str, VectorRecord] = {}

    async def upsert(self, vectors: list[VectorRecord]) -> None:
        for v in vectors:
            self._store[v["id"]] = v

    async def query(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        candidates: list[tuple[float, VectorRecord]] = []
        for rec in self._store.values():
            if metadata_filter:
                if not all(rec["metadata"].get(k) == v for k, v in metadata_filter.items()):
                    continue
            score = _cosine(vector, rec["values"])
            candidates.append((score, rec))

        candidates.sort(key=lambda t: t[0], reverse=True)
        return [
            QueryResult(id=rec["id"], score=score, metadata=rec["metadata"])
            for score, rec in candidates[:top_k]
        ]

    async def delete_all(self) -> None:
        self._store.clear()
