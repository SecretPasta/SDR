"""Tests for GeminiEmbedder — mocks the google-genai SDK at the boundary."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.adapters.gemini_embedder import GeminiEmbedder
from app.config import GeminiSettings

_BATCH_SIZE = 100


def _settings() -> GeminiSettings:
    return GeminiSettings(api_key="gemini-test-key", embed_dimensions=768)  # type: ignore[arg-type]


def _mock_embedding_response(n: int) -> MagicMock:
    embeddings = []
    for i in range(n):
        e = MagicMock()
        e.values = [float(i)] * 768
        embeddings.append(e)
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


class TestTaskTypes:
    def _capture_task_type(self, mock_client_inst: MagicMock) -> list[str]:
        """Extract task_type values from all embed_content calls."""
        task_types = []
        for c in mock_client_inst.aio.models.embed_content.call_args_list:
            config = c.kwargs.get("config") or (c.args[2] if len(c.args) > 2 else None)
            if config is not None:
                task_types.append(config.task_type)
        return task_types

    async def test_embed_documents_uses_retrieval_document_task_type(self) -> None:
        with patch("app.adapters.gemini_embedder.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.embed_content = AsyncMock(
                return_value=_mock_embedding_response(2)
            )
            emb = GeminiEmbedder(_settings())
            await emb.embed_documents(["text1", "text2"])

        task_types = self._capture_task_type(client_inst)
        assert all(t == "RETRIEVAL_DOCUMENT" for t in task_types)

    async def test_embed_query_uses_retrieval_query_task_type(self) -> None:
        with patch("app.adapters.gemini_embedder.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.embed_content = AsyncMock(
                return_value=_mock_embedding_response(1)
            )
            emb = GeminiEmbedder(_settings())
            await emb.embed_query("a query")

        task_types = self._capture_task_type(client_inst)
        assert all(t == "RETRIEVAL_QUERY" for t in task_types)

    async def test_embed_for_similarity_uses_semantic_similarity_task_type(self) -> None:
        with patch("app.adapters.gemini_embedder.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.embed_content = AsyncMock(
                return_value=_mock_embedding_response(1)
            )
            emb = GeminiEmbedder(_settings())
            await emb.embed_for_similarity(["heading"])

        task_types = self._capture_task_type(client_inst)
        assert all(t == "SEMANTIC_SIMILARITY" for t in task_types)


class TestBatching:
    async def test_inputs_over_batch_limit_split_into_multiple_calls(self) -> None:
        n_texts = _BATCH_SIZE + 5  # 105 texts

        with patch("app.adapters.gemini_embedder.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            # First batch: 100 embeddings; second batch: 5
            client_inst.aio.models.embed_content = AsyncMock(
                side_effect=[
                    _mock_embedding_response(_BATCH_SIZE),
                    _mock_embedding_response(5),
                ]
            )
            emb = GeminiEmbedder(_settings())
            results = await emb.embed_documents(["text"] * n_texts)

        assert client_inst.aio.models.embed_content.call_count == 2
        assert len(results) == n_texts

    async def test_empty_input_returns_empty_list_without_api_call(self) -> None:
        with patch("app.adapters.gemini_embedder.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            emb = GeminiEmbedder(_settings())
            results = await emb.embed_documents([])

        assert results == []
        client_inst.aio.models.embed_content.assert_not_called()

    async def test_embed_query_returns_single_vector(self) -> None:
        with patch("app.adapters.gemini_embedder.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.embed_content = AsyncMock(
                return_value=_mock_embedding_response(1)
            )
            emb = GeminiEmbedder(_settings())
            result = await emb.embed_query("query text")

        assert isinstance(result, list)
        assert len(result) == 768
