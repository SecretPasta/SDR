"""Tests for POST /chat/single and POST /chat/cross endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.domain.chat import ChatAnswer, Citation


def _fake_answer(text: str = "Answer text.", insufficient: bool = False) -> ChatAnswer:
    return ChatAnswer(
        answer=text,
        citations=[Citation(filename="doc.pdf", section="3.1", page=4)],
        insufficient_context=insufficient,
    )


@pytest.fixture
def fake_pipeline():
    pipeline = MagicMock()
    pipeline.answer = AsyncMock(return_value=_fake_answer())
    return pipeline


@pytest.fixture
def app_with_overrides(fake_pipeline):
    from app.main import app
    from app.deps import get_chat_pipeline
    app.dependency_overrides[get_chat_pipeline] = lambda: fake_pipeline
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def client(app_with_overrides):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_overrides), base_url="http://test"
    ) as c:
        yield c


class TestChatSingle:
    async def test_returns_200_with_answer_and_citations(self, client) -> None:
        resp = await client.post("/chat/single", json={"query": "What are the stages?", "doc_id": "A"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Answer text."
        assert isinstance(data["citations"], list)

    async def test_missing_doc_id_returns_422(self, client) -> None:
        resp = await client.post("/chat/single", json={"query": "Some question"})
        assert resp.status_code == 422

    async def test_empty_query_returns_422(self, client) -> None:
        resp = await client.post("/chat/single", json={"query": "", "doc_id": "A"})
        assert resp.status_code == 422

    async def test_pipeline_called_with_single_mode(self, client, fake_pipeline) -> None:
        await client.post("/chat/single", json={"query": "Stages?", "doc_id": "A"})

        fake_pipeline.answer.assert_called_once()
        _, kwargs = fake_pipeline.answer.call_args
        assert kwargs.get("mode") == "single" or fake_pipeline.answer.call_args.args[1] == "single"

    async def test_citations_formatted_as_strings(self, client) -> None:
        resp = await client.post("/chat/single", json={"query": "Stages?", "doc_id": "A"})

        data = resp.json()
        assert all(isinstance(c, str) for c in data["citations"])
        assert "doc.pdf · §3.1 · page 4" in data["citations"]

    async def test_insufficient_context_flag_propagated(self, client, fake_pipeline) -> None:
        fake_pipeline.answer = AsyncMock(return_value=_fake_answer(insufficient=True))

        resp = await client.post("/chat/single", json={"query": "Unknown topic", "doc_id": "A"})

        assert resp.json()["insufficient_context"] is True


class TestChatCross:
    async def test_returns_200_with_answer(self, client) -> None:
        resp = await client.post("/chat/cross", json={"query": "How did pricing change?"})

        assert resp.status_code == 200
        assert resp.json()["answer"] == "Answer text."

    async def test_pipeline_called_with_cross_mode(self, client, fake_pipeline) -> None:
        await client.post("/chat/cross", json={"query": "Compare the two versions."})

        fake_pipeline.answer.assert_called_once()
        call_args = fake_pipeline.answer.call_args
        assert "cross" in call_args.args or call_args.kwargs.get("mode") == "cross"

    async def test_doc_id_not_required_for_cross(self, client) -> None:
        resp = await client.post("/chat/cross", json={"query": "What changed?"})
        assert resp.status_code == 200

    async def test_empty_query_returns_422(self, client) -> None:
        resp = await client.post("/chat/cross", json={"query": ""})
        assert resp.status_code == 422
