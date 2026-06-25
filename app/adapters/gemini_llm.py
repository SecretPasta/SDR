from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import GeminiSettings

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, errors.ServerError):
        return True
    # 429 RESOURCE_EXHAUSTED comes back as ClientError
    if isinstance(exc, errors.ClientError):
        return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
    return False


class GeminiClient:
    def __init__(self, settings: GeminiSettings) -> None:
        self._client = genai.Client(api_key=settings.api_key.get_secret_value())
        self._model = settings.chat_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        contents = _to_contents(messages)

        if response_schema is not None:
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            )
        else:
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
            )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        text = response.text
        if text is None:
            raise ValueError("Gemini returned an empty response")
        if response_schema is not None:
            return json.loads(text)
        return {"content": text}


def _to_contents(messages: list[dict[str, str]]) -> list[types.Content]:
    _role = {"assistant": "model"}
    return [
        types.Content(
            role=_role.get(msg["role"], msg["role"]),
            parts=[types.Part.from_text(text=msg["content"])],
        )
        for msg in messages
    ]