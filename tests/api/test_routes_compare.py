"""Tests for POST /compare and GET /summary endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.domain.comparison import ComparisonResult, DiffEntry
from app.domain.summary import ExecutiveSummary, ImportantChange


@pytest.fixture(autouse=True)
def reset_compare_cache():
    """Reset module-level cache between tests."""
    import app.api.routes_compare as rc
    rc._cached_result = None
    rc._cached_summary = None
    yield
    rc._cached_result = None
    rc._cached_summary = None


@pytest.fixture
def fake_result() -> ComparisonResult:
    return ComparisonResult(
        diff=[DiffEntry(docA_text="old", docB_text="new", reason="updated",
                        sourceA="A.pdf · §1.0", sourceB="B.docx · §1.0")],
    )


@pytest.fixture
def fake_summary() -> ExecutiveSummary:
    return ExecutiveSummary(
        top_changes=[ImportantChange(rank=1, verdict="DIFF", summary="Pricing changed",
                                     why_it_matters="billing impact", citations=[])],
        total_matches=2, total_diffs=1, total_missing=0,
    )


@pytest.fixture
def fake_pipeline(fake_result, fake_summary):
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=(fake_result, fake_summary))
    return pipeline


@pytest.fixture
def app_with_overrides(fake_pipeline):
    from app.main import app
    from app.deps import get_comparison_pipeline
    app.dependency_overrides[get_comparison_pipeline] = lambda: fake_pipeline
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def client(app_with_overrides):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_overrides), base_url="http://test"
    ) as c:
        yield c


class TestPostCompare:
    async def test_returns_200_with_result_and_summary(self, client, tmp_path) -> None:
        pdf = tmp_path / "v0.pdf"
        docx = tmp_path / "v5.docx"
        pdf.write_bytes(b"fake")
        docx.write_bytes(b"fake")

        with (
            patch("app.api.routes_compare.parse_pdf", return_value=MagicMock(sections=[])),
            patch("app.api.routes_compare.parse_docx", return_value=MagicMock(sections=[])),
        ):
            resp = await client.post("/compare", json={
                "pdf_path": str(pdf), "docx_path": str(docx)
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        assert "summary" in data

    async def test_missing_pdf_returns_422(self, client, tmp_path) -> None:
        docx = tmp_path / "v5.docx"
        docx.write_bytes(b"fake")

        resp = await client.post("/compare", json={
            "pdf_path": "/nonexistent/file.pdf",
            "docx_path": str(docx),
        })

        assert resp.status_code == 422

    async def test_missing_docx_returns_422(self, client, tmp_path) -> None:
        pdf = tmp_path / "v0.pdf"
        pdf.write_bytes(b"fake")

        resp = await client.post("/compare", json={
            "pdf_path": str(pdf),
            "docx_path": "/nonexistent/file.docx",
        })

        assert resp.status_code == 422

    async def test_compare_populates_summary_cache(self, client, tmp_path) -> None:
        import app.api.routes_compare as rc

        pdf = tmp_path / "v0.pdf"
        docx = tmp_path / "v5.docx"
        pdf.write_bytes(b"x")
        docx.write_bytes(b"x")

        with (
            patch("app.api.routes_compare.parse_pdf", return_value=MagicMock(sections=[])),
            patch("app.api.routes_compare.parse_docx", return_value=MagicMock(sections=[])),
        ):
            await client.post("/compare", json={
                "pdf_path": str(pdf), "docx_path": str(docx)
            })

        assert rc._cached_summary is not None


class TestGetSummary:
    async def test_returns_404_when_no_comparison_run(self, client) -> None:
        resp = await client.get("/summary")
        assert resp.status_code == 404

    async def test_returns_200_after_compare(self, client, tmp_path) -> None:
        pdf = tmp_path / "v0.pdf"
        docx = tmp_path / "v5.docx"
        pdf.write_bytes(b"x")
        docx.write_bytes(b"x")

        with (
            patch("app.api.routes_compare.parse_pdf", return_value=MagicMock(sections=[])),
            patch("app.api.routes_compare.parse_docx", return_value=MagicMock(sections=[])),
        ):
            await client.post("/compare", json={
                "pdf_path": str(pdf), "docx_path": str(docx)
            })

        resp = await client.get("/summary")
        assert resp.status_code == 200
        assert "summary" in resp.json()
