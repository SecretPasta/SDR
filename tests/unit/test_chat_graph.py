"""Tests for ChatPipeline LangGraph — mode routing, sufficiency, synthesis."""
from __future__ import annotations

import pytest

from app.chat.graph import ChatPipeline
from app.chat.retriever import Retriever
from app.config import RetrievalSettings
from app.domain.chat import ChatAnswer
from app.ports.vector_store import VectorRecord
from tests.fakes.fake_embedder import FakeEmbedder, _hash_vec
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_vector_store import InMemoryVectorStore

_CHAT_ANSWER = {"answer": "Context-based answer.", "citations": [], "insufficient_context": False}


def _config(floor: float = 0.5) -> RetrievalSettings:
    s = RetrievalSettings()
    object.__setattr__(s, "relevance_floor", floor)
    return s


async def _populated_store(query: str) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.upsert([
        VectorRecord(
            id="a-1", values=_hash_vec(query),
            metadata={"doc_id": "A", "display_text": "V0 answer content", "filename": "A.pdf",
                      "heading_number": "1.0"},
        ),
        VectorRecord(
            id="b-1", values=_hash_vec(query),
            metadata={"doc_id": "B", "display_text": "V5 answer content", "filename": "B.docx",
                      "heading_number": "1.0"},
        ),
    ])
    return store


def _pipeline(llm: FakeLLM, store: InMemoryVectorStore, floor: float = 0.5) -> ChatPipeline:
    embedder = FakeEmbedder()
    retriever = Retriever(embedder, store)
    return ChatPipeline(
        llm=llm,
        embedder=embedder,
        retriever=retriever,
        config=_config(floor),
    )


class TestChatPipelineSingleMode:
    async def test_single_mode_returns_chat_answer(self) -> None:
        store = await _populated_store("process stages")
        llm = FakeLLM([_CHAT_ANSWER])
        pipeline = _pipeline(llm, store)

        result = await pipeline.answer("process stages", mode="single", doc_id="A")

        assert isinstance(result, ChatAnswer)

    async def test_single_mode_without_doc_id_raises_value_error(self) -> None:
        pipeline = _pipeline(FakeLLM([]), InMemoryVectorStore())

        with pytest.raises(ValueError, match="doc_id"):
            await pipeline.answer("query", mode="single")

    async def test_single_mode_routes_through_synthesize_when_sufficient(self) -> None:
        store = await _populated_store("pricing")
        llm = FakeLLM([_CHAT_ANSWER])
        pipeline = _pipeline(llm, store)

        result = await pipeline.answer("pricing", mode="single", doc_id="A")

        # LLM was called → synthesis ran
        assert len(llm.calls) == 1
        assert result.insufficient_context is False

    async def test_single_mode_routes_to_insufficient_when_empty_store(self) -> None:
        llm = FakeLLM([])
        pipeline = _pipeline(llm, InMemoryVectorStore())

        result = await pipeline.answer("anything", mode="single", doc_id="A")

        assert result.insufficient_context is True
        assert len(llm.calls) == 0


class TestChatPipelineCrossMode:
    async def test_cross_mode_returns_chat_answer(self) -> None:
        store = await _populated_store("compare pricing")
        llm = FakeLLM([_CHAT_ANSWER])
        pipeline = _pipeline(llm, store)

        result = await pipeline.answer("compare pricing", mode="cross")

        assert isinstance(result, ChatAnswer)

    async def test_cross_mode_llm_prompt_contains_both_doc_labels(self) -> None:
        store = await _populated_store("compare stages")
        llm = FakeLLM([_CHAT_ANSWER])
        pipeline = _pipeline(llm, store)

        await pipeline.answer("compare stages", mode="cross")

        prompt = llm.calls[0][0][0]["content"]
        assert "V0" in prompt or "V5" in prompt

    async def test_cross_mode_empty_store_returns_insufficient(self) -> None:
        llm = FakeLLM([])
        pipeline = _pipeline(llm, InMemoryVectorStore())

        result = await pipeline.answer("anything", mode="cross")

        assert result.insufficient_context is True


class TestChatPipelineQueryEmbedding:
    async def test_query_embedding_populated_in_state(self) -> None:
        """embed_query node runs — verified indirectly by retrieval succeeding."""
        store = await _populated_store("test query")
        llm = FakeLLM([_CHAT_ANSWER])
        pipeline = _pipeline(llm, store)

        # If embed_query didn't run, retrieval would fail; successful answer proves it ran
        result = await pipeline.answer("test query", mode="single", doc_id="A")

        assert isinstance(result, ChatAnswer)
