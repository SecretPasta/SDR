"""Tests for ClaudeClient — mocks the anthropic SDK at the boundary."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.adapters.claude_llm import ClaudeClient
from app.config import AnthropicSettings


def _settings() -> AnthropicSettings:
    return AnthropicSettings(api_key="sk-ant-test", model="claude-sonnet-4-6")  # type: ignore[arg-type]


class _Schema(BaseModel):
    verdict: str
    reason: str


def _text_response(text: str) -> MagicMock:
    """Build a mock Message with a single TextBlock."""
    from anthropic.types import TextBlock
    msg = MagicMock()
    msg.content = [TextBlock(type="text", text=text)]
    return msg


def _tool_response(tool_name: str, input_data: dict) -> MagicMock:
    """Build a mock Message with a single ToolUseBlock."""
    from anthropic.types import ToolUseBlock
    block = ToolUseBlock(type="tool_use", id="call_123", name=tool_name, input=input_data)
    msg = MagicMock()
    msg.content = [block]
    return msg


class TestUnstructuredGenerate:
    async def test_returns_content_key_with_text(self) -> None:
        mock_response = _text_response("Hello from Claude.")

        with patch("app.adapters.claude_llm.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create = AsyncMock(return_value=mock_response)
            client = ClaudeClient(_settings())

            result = await client.generate([{"role": "user", "content": "Hi"}])

        assert result == {"content": "Hello from Claude."}

    async def test_api_key_passed_to_async_anthropic(self) -> None:
        with patch("app.adapters.claude_llm.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create = AsyncMock(return_value=_text_response("ok"))
            ClaudeClient(_settings())

        MockClient.assert_called_once_with(api_key="sk-ant-test")


class TestStructuredGenerate:
    async def test_structured_output_uses_tool_use(self) -> None:
        tool_input = {"verdict": "MATCH", "reason": "same"}
        mock_response = _tool_response("_Schema", tool_input)

        with patch("app.adapters.claude_llm.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create = AsyncMock(return_value=mock_response)
            client = ClaudeClient(_settings())

            result = await client.generate(
                [{"role": "user", "content": "judge this"}],
                response_schema=_Schema,
            )

        assert result == tool_input

    async def test_structured_call_includes_tools_in_request(self) -> None:
        tool_input = {"verdict": "DIFF", "reason": "changed"}
        mock_response = _tool_response("_Schema", tool_input)

        with patch("app.adapters.claude_llm.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create = AsyncMock(return_value=mock_response)
            client = ClaudeClient(_settings())

            await client.generate(
                [{"role": "user", "content": "judge"}],
                response_schema=_Schema,
            )

        call_kwargs = instance.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_choice" in call_kwargs


class TestRetryBehavior:
    async def test_retries_on_connection_error_and_succeeds(self) -> None:
        import httpx
        import anthropic

        good_response = _text_response("ok after retry")
        request = httpx.Request("POST", "https://api.anthropic.com")
        error = anthropic.APIConnectionError(request=request)

        with patch("app.adapters.claude_llm.AsyncAnthropic") as MockClient:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                instance = MockClient.return_value
                instance.messages.create = AsyncMock(
                    side_effect=[error, error, good_response]
                )
                client = ClaudeClient(_settings())

                result = await client.generate([{"role": "user", "content": "hi"}])

        assert result["content"] == "ok after retry"
        assert instance.messages.create.call_count == 3

    async def test_is_transient_returns_true_for_rate_limit(self) -> None:
        from app.adapters.claude_llm import _is_transient
        import anthropic
        import httpx

        response = httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com"))
        exc = anthropic.RateLimitError("rate limit", response=response, body={})
        assert _is_transient(exc) is True

    async def test_is_transient_returns_false_for_auth_error(self) -> None:
        from app.adapters.claude_llm import _is_transient
        import anthropic
        import httpx

        response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com"))
        exc = anthropic.AuthenticationError("auth", response=response, body={})
        assert _is_transient(exc) is False
