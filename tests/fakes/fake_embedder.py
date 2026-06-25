"""FakeEmbedder — deterministic EmbedderClient for unit tests."""
from __future__ import annotations

import hashlib
import math


def _hash_vec(text: str, dim: int = 768) -> list[float]:
    """Deterministic unit-length vector from text via SHA-256 expansion."""
    raw = hashlib.sha256(text.encode()).digest()
    # Repeat the 32-byte digest to fill dim floats, then normalise
    extended = (raw * (dim // len(raw) + 1))[:dim]
    vec = [float(b) - 128.0 for b in extended]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class FakeEmbedder:
    """Implements EmbedderClient protocol with hash-based deterministic embeddings.

    Same input always → same output (critical for alignment tests that check determinism).
    All three embed methods behave identically (we test logic, not provider differences).
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self._dim) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _hash_vec(text, self._dim)

    async def embed_for_similarity(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self._dim) for t in texts]
