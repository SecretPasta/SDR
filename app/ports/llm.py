from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: type | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...