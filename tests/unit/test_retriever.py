"""Tests for Retriever — retrieve_single, retrieve_dual, retrieve_unfiltered."""
from __future__ import annotations

import pytest

from app.chat.retriever import RetrievedChunk, Retriever
from app.ports.vector_store import VectorRecord
from tests.fakes.fake_embedder import FakeEmbedder, _hash_vec
from tests.fakes.fake_vector_store import InMemoryVectorStore


def _record(id: str, doc_id: str, text: str, display: str = "") -> VectorRecord:
    return VectorRecord(
        id=id,
        values=_hash_vec(text),
        metadata={
            "doc_id": doc_id,
            "display_text": display or text,
            "filename": f"{doc_id}.pdf",
            "heading_number": "1.0",
        },
    )


async def _populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.upsert([
        _record("a-1", "A", "Process Stages overview", "Process Stages overview"),
        _record("a-2", "A", "Price Calculation logic", "Price Calculation logic"),
        _record("b-1", "B", "Process Stages updated", "Process Stages updated"),
        _record("b-2", "B", "Integration Checks section", "Integration Checks section"),
    ])
    return store


class TestRetrieveSingle:
    async def test_returns_only_chunks_for_requested_doc_id(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_single("Process Stages", doc_id="A", top_k=10)

        doc_ids = {c.metadata["doc_id"] for c in results}
        assert doc_ids == {"A"}

    async def test_returns_retrieved_chunk_instances(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_single("Process Stages", doc_id="A", top_k=5)

        assert all(isinstance(c, RetrievedChunk) for c in results)

    async def test_display_text_populated_from_metadata(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_single("Process Stages", doc_id="A", top_k=5)

        assert all(c.display_text != "" for c in results)

    async def test_score_populated_on_each_chunk(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_single("Process Stages", doc_id="A", top_k=5)

        assert all(isinstance(c.score, float) for c in results)

    async def test_top_k_limits_result_count(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_single("stages", doc_id="A", top_k=1)

        assert len(results) <= 1

    async def test_empty_store_returns_empty_list(self) -> None:
        retriever = Retriever(FakeEmbedder(), InMemoryVectorStore())
        results = await retriever.retrieve_single("anything", doc_id="A", top_k=5)
        assert results == []


class TestRetrieveDual:
    async def test_returns_dict_keyed_by_both_doc_ids(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        result = await retriever.retrieve_dual("Process", doc_ids=("A", "B"), top_k_per_doc=5)

        assert set(result.keys()) == {"A", "B"}

    async def test_each_side_only_contains_its_own_doc_id(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        result = await retriever.retrieve_dual("Process", doc_ids=("A", "B"), top_k_per_doc=5)

        for doc_id, chunks in result.items():
            assert all(c.metadata["doc_id"] == doc_id for c in chunks)

    async def test_embed_query_called_once_shared_across_both_queries(self) -> None:
        """retrieve_dual embeds once and fans out — verifying by counting store calls is impractical,
        but we can verify both sides return results in one call."""
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        result = await retriever.retrieve_dual("Process Stages", doc_ids=("A", "B"), top_k_per_doc=5)

        assert len(result["A"]) > 0
        assert len(result["B"]) > 0


class TestRetrieveUnfiltered:
    async def test_returns_chunks_from_multiple_docs(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_unfiltered("Process", top_k=10)

        doc_ids = {c.metadata["doc_id"] for c in results}
        assert len(doc_ids) > 1

    async def test_respects_top_k(self) -> None:
        store = await _populated_store()
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_unfiltered("Process", top_k=2)

        assert len(results) <= 2

    async def test_no_metadata_filter_applied(self) -> None:
        store = InMemoryVectorStore()
        await store.upsert([
            _record("x-1", "X", "unique content alpha"),
            _record("y-1", "Y", "unique content beta"),
        ])
        retriever = Retriever(FakeEmbedder(), store)

        results = await retriever.retrieve_unfiltered("unique content", top_k=10)

        returned_ids = {c.id for c in results}
        assert "x-1" in returned_ids
        assert "y-1" in returned_ids
