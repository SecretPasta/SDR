from __future__ import annotations

import logging

from app.chat.retriever import RetrievedChunk, Retriever
from app.config import RetrievalSettings
from app.domain.chat import ChatAnswer
from app.ports.llm import LLMClient
from app.prompts.chat_synthesis import build_cross_doc_prompt

logger = logging.getLogger(__name__)

_FALLBACK_TOP_K = 6


def _max_score(chunks: list[RetrievedChunk]) -> float:
    return max((c.score for c in chunks), default=0.0)


class CrossDocChat:
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        config: RetrievalSettings,
        doc_ids: tuple[str, str] = ("A", "B"),
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._config = config
        self._doc_ids = doc_ids

    async def answer(self, query: str) -> ChatAnswer:
        chunks_by_doc = await self._retriever.retrieve_dual(
            query,
            doc_ids=self._doc_ids,
            top_k_per_doc=self._config.cross_doc_top_k,
        )

        floor = self._config.relevance_floor
        both_below_floor = all(
            _max_score(chunks_by_doc.get(did, [])) < floor
            for did in self._doc_ids
        )

        if both_below_floor:
            logger.info(
                "Both sides below relevance floor (%.2f); attempting unfiltered fallback",
                floor,
            )
            chunks_by_doc = await self._fallback(query, floor)
            if chunks_by_doc is None:
                return ChatAnswer(
                    answer="I couldn't find relevant information in either document to answer this question.",
                    insufficient_context=True,
                )

        prompt = build_cross_doc_prompt(query, chunks_by_doc)
        logger.debug(
            "Cross-doc prompt built (%s)",
            ", ".join(f"{did}={len(chunks_by_doc.get(did, []))} chunks" for did in self._doc_ids),
        )

        raw = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            response_schema=ChatAnswer,
        )
        return ChatAnswer.model_validate(raw)

    async def _fallback(
        self,
        query: str,
        floor: float,
    ) -> dict[str, list[RetrievedChunk]] | None:
        """Unfiltered recovery query; returns None if still below the relevance floor."""
        all_chunks = await self._retriever.retrieve_unfiltered(query, top_k=_FALLBACK_TOP_K)

        if _max_score(all_chunks) < floor:
            logger.info("Unfiltered fallback also below relevance floor — giving up")
            return None

        # Group by doc_id from metadata so the prompt builder gets labeled blocks
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in all_chunks:
            did = chunk.metadata.get("doc_id", "unknown")
            grouped.setdefault(did, []).append(chunk)

        return grouped
