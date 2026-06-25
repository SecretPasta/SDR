"""FakeLLM — deterministic LLMClient for unit tests."""
from __future__ import annotations

from typing import Any, Callable


class FakeLLM:
    """Implements LLMClient protocol with a canned-response queue or callable handler.

    Pass a list of dicts to pop in order, or a callable ``handler(messages, schema) -> dict``
    for schema-aware dispatch. All calls are recorded in ``self.calls``.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        handler: Callable[[list[dict], Any], dict[str, Any]] | None = None,
    ) -> None:
        self._queue: list[dict[str, Any]] = list(responses or [])
        self._handler = handler
        self.calls: list[tuple[list[dict], Any]] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        response_schema: type | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        self.calls.append((messages, response_schema))

        if self._handler is not None:
            return self._handler(messages, response_schema)

        if not self._queue:
            raise RuntimeError(
                f"FakeLLM queue exhausted after {len(self.calls)} call(s). "
                "Pass more canned responses or a handler."
            )
        return self._queue.pop(0)
