"""Tests for CrossDocChat — dual retrieval, fallback path, synthesis."""
from __future__ import annotations

import pytest

from app.chat.cross_doc import CrossDocChat
from app.chat.retriever import Retriever
from app.config import RetrievalSettings
from app.domain.chat import ChatAnswer
from app.ports.vector_store import VectorRecord
from tests.fakes.fake_embedder import FakeEmbedder, _hash_vec
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_vector_store import InMemoryVectorStore

_CHAT_ANSWER = {"answer": "V5 adds discount logic.", "citations": [], "insufficient_context": False}


def _config(floor: float = 0.5) -> RetrievalSettings:
    s = RetrievalSettings()
    object.__setattr__(s, "relevance_floor", floor)
    return s


async def _dual_store(query: str) -> InMemoryVectorStore:
    """One high-score chunk per doc, both matching the query."""
    store = InMemoryVectorStore()
    await store.upsert([
        VectorRecord(
            id="a-1", values=_hash_vec(query),
            metadata={"doc_id": "A", "display_text": "V0 content", "filename": "FDS_V0.pdf"},
        ),
        VectorRecord(
            id="b-1", values=_hash_vec(query),
            metadata={"doc_id": "B", "display_text": "V5 content", "filename": "FDS_V5.docx"},
        ),
    ])
    return store


class TestCrossDocChatHappyPath:
    async def test_returns_chat_answer_instance(self) -> None:
        store = await _dual_store("pricing query")
        llm = FakeLLM([_CHAT_ANSWER])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), store), _config())

        result = await chat.answer("pricing query")

        assert isinstance(result, ChatAnswer)

    async def test_llm_called_with_chat_answer_schema(self) -> None:
        store = await _dual_store("stages query")
        llm = FakeLLM([_CHAT_ANSWER])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), store), _config())

        await chat.answer("stages query")

        _, schema = llm.calls[0]
        assert schema is ChatAnswer

    async def test_llm_receives_prompt_with_both_doc_labels(self) -> None:
        store = await _dual_store("stages query")
        llm = FakeLLM([_CHAT_ANSWER])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), store), _config())

        await chat.answer("stages query")

        prompt = llm.calls[0][0][0]["content"]
        assert "V0" in prompt
        assert "V5" in prompt


class TestCrossDocChatFallback:
    async def test_both_below_floor_triggers_unfiltered_fallback(self) -> None:
        # Store chunks for A and B with unrelated text → low cosine with query
        store = InMemoryVectorStore()
        unrelated = _hash_vec("zzzzz totally unrelated zzzzz")
        await store.upsert([
            VectorRecord(id="a-1", values=unrelated,
                         metadata={"doc_id": "A", "display_text": "A", "filename": "A.pdf"}),
            VectorRecord(id="b-1", values=unrelated,
                         metadata={"doc_id": "B", "display_text": "B", "filename": "B.docx"}),
        ])

        # Now add a high-score chunk with no doc_id filter — will be hit by unfiltered query
        await store.upsert([
            VectorRecord(id="fallback-1", values=_hash_vec("pricing query"),
                         metadata={"doc_id": "A", "display_text": "fallback result", "filename": "A.pdf"}),
        ])

        config = _config(floor=0.5)
        llm = FakeLLM([_CHAT_ANSWER])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), store), config)

        result = await chat.answer("pricing query")

        # LLM should have been called (fallback succeeded)
        assert len(llm.calls) == 1
        assert isinstance(result, ChatAnswer)

    async def test_both_below_floor_and_fallback_fails_returns_insufficient(self) -> None:
        # All stored vectors are unrelated to the query
        store = InMemoryVectorStore()
        unrelated = _hash_vec("zzzzz totally unrelated content zzzzz")
        await store.upsert([
            VectorRecord(id="a-1", values=unrelated,
                         metadata={"doc_id": "A", "display_text": "A", "filename": "A.pdf"}),
        ])
        config = _config(floor=0.99)  # very high floor — nothing will pass
        llm = FakeLLM([])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), store), config)

        result = await chat.answer("pricing model details")

        assert result.insufficient_context is True
        assert len(llm.calls) == 0

    async def test_empty_store_returns_insufficient_context(self) -> None:
        llm = FakeLLM([])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), InMemoryVectorStore()), _config())

        result = await chat.answer("anything")

        assert result.insufficient_context is True

    async def test_fallback_groups_results_by_doc_id(self) -> None:
        """Unfiltered results get re-labeled per their metadata doc_id in the prompt."""
        store = InMemoryVectorStore()
        unrelated = _hash_vec("unrelated zz")
        query = "compare pricing"
        await store.upsert([
            VectorRecord(id="a-1", values=unrelated,
                         metadata={"doc_id": "A", "display_text": "A", "filename": "A.pdf"}),
            VectorRecord(id="b-1", values=unrelated,
                         metadata={"doc_id": "B", "display_text": "B", "filename": "B.docx"}),
            VectorRecord(id="fb-a", values=_hash_vec(query),
                         metadata={"doc_id": "A", "display_text": "V0 pricing detail", "filename": "A.pdf"}),
            VectorRecord(id="fb-b", values=_hash_vec(query),
                         metadata={"doc_id": "B", "display_text": "V5 pricing detail", "filename": "B.docx"}),
        ])
        config = _config(floor=0.5)
        llm = FakeLLM([_CHAT_ANSWER])
        chat = CrossDocChat(llm, Retriever(FakeEmbedder(), store), config)

        await chat.answer(query)

        prompt = llm.calls[0][0][0]["content"]
        assert "V0 pricing detail" in prompt or "V5 pricing detail" in prompt
