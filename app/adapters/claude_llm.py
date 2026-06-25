from __future__ import annotations

import logging
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolChoiceToolParam,
    ToolParam,
    ToolUseBlock,
)
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import AnthropicSettings

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


class ClaudeClient:
    def __init__(self, settings: AnthropicSettings) -> None:
        self._client = AsyncAnthropic(api_key=settings.api_key.get_secret_value())
        self._model = settings.model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[MessageParam],
        *,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        if response_schema is not None:
            return await self._generate_structured(messages, response_schema, max_tokens)
        return await self._generate_unstructured(messages, max_tokens)

    async def _generate_unstructured(
        self,
        messages: list[MessageParam],
        max_tokens: int,
    ) -> dict[str, Any]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        text_block = next(b for b in response.content if isinstance(b, TextBlock))
        return {"content": text_block.text}

    async def _generate_structured(
        self,
        messages: list[MessageParam],
        response_schema: type[BaseModel],
        max_tokens: int,
    ) -> dict[str, Any]:
        tool_name = response_schema.__name__
        tool: ToolParam = {
            "name": tool_name,
            "description": f"Respond using the {tool_name} schema.",
            "input_schema": response_schema.model_json_schema(),  # type: ignore[typeddict-item]
        }
        tool_choice: ToolChoiceToolParam = {"type": "tool", "name": tool_name}
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice,
        )
        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == tool_name:
                return block.input  # type: ignore[return-value]
        raise ValueError(f"Anthropic returned no tool_use block for '{tool_name}'")
