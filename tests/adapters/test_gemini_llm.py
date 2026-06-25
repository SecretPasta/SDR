"""Tests for GeminiClient — mocks the google-genai SDK at the boundary."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.adapters.gemini_llm import GeminiClient, _to_contents
from app.config import GeminiSettings


def _settings() -> GeminiSettings:
    return GeminiSettings(api_key="gemini-test-key")  # type: ignore[arg-type]


class _Schema(BaseModel):
    answer: str


def _mock_response(text: str) -> MagicMock:
    part = MagicMock()
    part.text = text
    part.thought = False
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


class TestUnstructuredGenerate:
    async def test_returns_content_key_with_text(self) -> None:
        with patch("app.adapters.gemini_llm.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.generate_content = AsyncMock(
                return_value=_mock_response("Gemini response")
            )

            client = GeminiClient(_settings())
            result = await client.generate([{"role": "user", "content": "Hello"}])

        assert result == {"content": "Gemini response"}

    async def test_api_key_passed_to_genai_client(self) -> None:
        with patch("app.adapters.gemini_llm.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.generate_content = AsyncMock(
                return_value=_mock_response("ok")
            )
            GeminiClient(_settings())

        mock_genai.Client.assert_called_once_with(api_key="gemini-test-key")


class TestStructuredGenerate:
    async def test_structured_call_sets_response_mime_type(self) -> None:
        import json

        with patch("app.adapters.gemini_llm.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.generate_content = AsyncMock(
                return_value=_mock_response(json.dumps({"answer": "42"}))
            )

            # Capture the config passed to generate_content
            client = GeminiClient(_settings())
            result = await client.generate(
                [{"role": "user", "content": "q"}],
                response_schema=_Schema,
            )

        assert result == {"answer": "42"}
        call_kwargs = client_inst.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is _Schema

    async def test_structured_response_parsed_as_json(self) -> None:
        import json

        with patch("app.adapters.gemini_llm.genai") as mock_genai:
            client_inst = MagicMock()
            mock_genai.Client.return_value = client_inst
            client_inst.aio.models.generate_content = AsyncMock(
                return_value=_mock_response(json.dumps({"answer": "hello"}))
            )
            client = GeminiClient(_settings())
            result = await client.generate(
                [{"role": "user", "content": "q"}],
                response_schema=_Schema,
            )

        assert result["answer"] == "hello"


class TestToContents:
    def test_user_role_maps_to_user(self) -> None:
        contents = _to_contents([{"role": "user", "content": "hi"}])
        assert contents[0].role == "user"

    def test_assistant_role_maps_to_model(self) -> None:
        contents = _to_contents([{"role": "assistant", "content": "ok"}])
        assert contents[0].role == "model"

    def test_text_content_wrapped_in_part(self) -> None:
        contents = _to_contents([{"role": "user", "content": "hello world"}])
        assert contents[0].parts[0].text == "hello world"


class TestRetryBehavior:
    async def test_retries_on_server_error_and_succeeds(self) -> None:
        from google.genai import errors

        with patch("app.adapters.gemini_llm.genai") as mock_genai:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                client_inst = MagicMock()
                mock_genai.Client.return_value = client_inst

                server_error = errors.ServerError("Internal error", {"error": {}})
                client_inst.aio.models.generate_content = AsyncMock(
                    side_effect=[server_error, server_error, _mock_response("ok")]
                )
                client = GeminiClient(_settings())
                result = await client.generate([{"role": "user", "content": "hi"}])

        assert result["content"] == "ok"
        assert client_inst.aio.models.generate_content.call_count == 3

    async def test_is_transient_true_for_server_error(self) -> None:
        from app.adapters.gemini_llm import _is_transient
        from google.genai import errors

        exc = errors.ServerError("Internal error", {"error": {}})
        assert _is_transient(exc) is True

    async def test_is_transient_false_for_generic_exception(self) -> None:
        from app.adapters.gemini_llm import _is_transient

        assert _is_transient(ValueError("oops")) is False
