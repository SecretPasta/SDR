from typing import Any, Protocol, TypedDict, runtime_checkable


class VectorRecord(TypedDict):
    id: str
    values: list[float]
    metadata: dict[str, Any]


class QueryResult(TypedDict):
    id: str
    score: float
    metadata: dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, vectors: list[VectorRecord]) -> None: ...

    async def query(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]: ...

    async def delete_all(self) -> None: ...