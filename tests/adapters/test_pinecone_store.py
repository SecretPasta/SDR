"""Tests for PineconeStore — mocks the Pinecone SDK at the boundary."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.adapters.pinecone_store import PineconeStore
from app.config import PineconeSettings
from app.ports.vector_store import VectorRecord


def _settings() -> PineconeSettings:
    return PineconeSettings(  # type: ignore[arg-type]
        api_key="pinecone-test-key",
        index_name="test-index",
        namespace="default",
    )


def _record(id: str, dim: int = 4) -> VectorRecord:
    return VectorRecord(id=id, values=[0.1] * dim, metadata={"doc_id": "A"})


def _make_idx_entry(name: str) -> MagicMock:
    """MagicMock(name=...) sets the mock's display name, not the .name attr."""
    m = MagicMock()
    m.name = name
    return m


def _setup_pinecone_mock(existing_indexes: list[str] | None = None):
    """Return (mock_pinecone_class, mock_pc_instance, mock_index)."""
    mock_pc = MagicMock()
    mock_index = MagicMock()

    mock_pc.list_indexes.return_value = [_make_idx_entry(n) for n in (existing_indexes or [])]
    mock_pc.Index.return_value = mock_index
    mock_index.upsert = MagicMock()
    mock_index.query = MagicMock()
    mock_index.delete = MagicMock()
    return mock_pc, mock_index


class TestIndexCreation:
    async def test_first_upsert_creates_index_when_absent(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=[])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.upsert([_record("v1")])

        mock_pc.create_index.assert_called_once()

    async def test_first_upsert_skips_creation_when_index_exists(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.upsert([_record("v1")])

        mock_pc.create_index.assert_not_called()

    async def test_subsequent_upserts_do_not_recreate_index(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=[])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.upsert([_record("v1")])
            await store.upsert([_record("v2")])

        # create_index called only once across two upserts
        assert mock_pc.create_index.call_count == 1


class TestUpsert:
    async def test_upsert_calls_index_upsert_with_correct_namespace(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.upsert([_record("v1"), _record("v2")])

        mock_index.upsert.assert_called_once()
        call_kwargs = mock_index.upsert.call_args.kwargs
        assert call_kwargs["namespace"] == "default"

    async def test_upsert_batches_at_100_records(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            records = [_record(f"v{i}") for i in range(250)]
            await store.upsert(records)

        # 250 records → 3 batches: 100, 100, 50
        assert mock_index.upsert.call_count == 3

    async def test_upsert_passes_id_values_metadata(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            rec = _record("my-id")
            await store.upsert([rec])

        vectors_arg = mock_index.upsert.call_args.kwargs["vectors"]
        assert len(vectors_arg) == 1
        assert vectors_arg[0]["id"] == "my-id"
        assert vectors_arg[0]["values"] == [0.1] * 4
        assert vectors_arg[0]["metadata"] == {"doc_id": "A"}


class TestQuery:
    async def test_query_forwards_filter_dict(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])
        mock_index.query.return_value = MagicMock(matches=[])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.query([0.1] * 4, top_k=5, metadata_filter={"doc_id": "A"})

        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs.get("filter") == {"doc_id": "A"}

    async def test_query_returns_list_of_query_results(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])
        match = MagicMock()
        match.id = "v1"
        match.score = 0.92
        match.metadata = {"doc_id": "A", "display_text": "hello"}
        mock_index.query.return_value = MagicMock(matches=[match])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            results = await store.query([0.1] * 4, top_k=3)

        assert len(results) == 1
        assert results[0]["id"] == "v1"
        assert results[0]["score"] == pytest.approx(0.92)

    async def test_query_with_no_filter_does_not_pass_filter_key(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])
        mock_index.query.return_value = MagicMock(matches=[])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.query([0.1] * 4, top_k=5)

        call_kwargs = mock_index.query.call_args.kwargs
        assert "filter" not in call_kwargs


class TestDeleteAll:
    async def test_delete_all_calls_index_delete_with_delete_all_true(self) -> None:
        mock_pc, mock_index = _setup_pinecone_mock(existing_indexes=["test-index"])

        with patch("app.adapters.pinecone_store.Pinecone", return_value=mock_pc):
            store = PineconeStore(_settings(), dimension=4)
            await store.delete_all()

        mock_index.delete.assert_called_once()
        call_kwargs = mock_index.delete.call_args.kwargs
        assert call_kwargs.get("delete_all") is True
        assert call_kwargs.get("namespace") == "default"
