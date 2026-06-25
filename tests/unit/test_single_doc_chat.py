"""Tests for SingleDocChat — retrieval + sufficiency check + synthesis."""
from __future__ import annotations

import pytest

from app.chat.retriever import Retriever
from app.chat.single_doc import SingleDocChat
from app.config import RetrievalSettings
from app.domain.chat import ChatAnswer
from app.ports.vector_store import VectorRecord
from tests.fakes.fake_embedder import FakeEmbedder, _hash_vec
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_vector_store import InMemoryVectorStore

_CHAT_ANSWER = {"answer": "The process has three stages.", "citations": [], "insufficient_context": False}
_SETTINGS = RetrievalSettings()


def _config(floor: float = 0.5) -> RetrievalSettings:
    s = RetrievalSettings()
    object.__setattr__(s, "relevance_floor", floor)
    return s


async def _store_with_chunk(doc_id: str, text: str) -> InMemoryVectorStore:
    """Populate a store with one chunk whose vector matches the query text exactly."""
    store = InMemoryVectorStore()
    await store.upsert([VectorRecord(
        id=f"{doc_id}-1",
        values=_hash_vec(text),
        metadata={
            "doc_id": doc_id,
            "display_text": text,
            "filename": f"{doc_id}.pdf",
            "heading_number": "1.0",
        },
    )])
    return store


class TestSingleDocChatHappyPath:
    async def test_returns_chat_answer_instance(self) -> None:
        store = await _store_with_chunk("A", "Process stages query")
        llm = FakeLLM([_CHAT_ANSWER])
        chat = SingleDocChat(llm, Retriever(FakeEmbedder(), store), _SETTINGS)

        result = await chat.answer("Process stages query", doc_id="A")

        assert isinstance(result, ChatAnswer)

    async def test_answer_text_comes_from_llm(self) -> None:
        store = await _store_with_chunk("A", "pricing query")
        llm = FakeLLM([{"answer": "Pricing is tier-based.", "citations": [], "insufficient_context": False}])
        chat = SingleDocChat(llm, Retriever(FakeEmbedder(), store), _SETTINGS)

        result = await chat.answer("pricing query", doc_id="A")

        assert result.answer == "Pricing is tier-based."

    async def test_llm_called_with_chat_answer_schema(self) -> None:
        store = await _store_with_chunk("A", "some query")
        llm = FakeLLM([_CHAT_ANSWER])
        chat = SingleDocChat(llm, Retriever(FakeEmbedder(), store), _SETTINGS)

        await chat.answer("some query", doc_id="A")

        _, schema = llm.calls[0]
        assert schema is ChatAnswer


class TestSingleDocChatInsufficientContext:
    async def test_empty_store_returns_insufficient_context(self) -> None:
        llm = FakeLLM([])
        chat = SingleDocChat(llm, Retriever(FakeEmbedder(), InMemoryVectorStore()), _SETTINGS)

        result = await chat.answer("anything", doc_id="A")

        assert result.insufficient_context is True
        assert len(llm.calls) == 0

    async def test_low_score_chunks_return_insufficient_context(self) -> None:
        # Store chunk for doc A but query with completely different text → low cosine
        store = await _store_with_chunk("A", "zzzzz totally unrelated document text zzzz")
        # Set a high floor so even moderate scores fail
        config = _config(floor=0.99)
        llm = FakeLLM([])
        chat = SingleDocChat(llm, Retriever(FakeEmbedder(), store), config)

        result = await chat.answer("Process stages", doc_id="A")

        assert result.insufficient_context is True
        assert len(llm.calls) == 0

    async def test_insufficient_response_has_non_empty_answer(self) -> None:
        llm = FakeLLM([])
        chat = SingleDocChat(llm, Retriever(FakeEmbedder(), InMemoryVectorStore()), _SETTINGS)

        result = await chat.answer("anything", doc_id="A")

        assert result.answer != ""
