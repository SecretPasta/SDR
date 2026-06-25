from __future__ import annotations

import logging

from app.chat.retriever import Retriever
from app.config import RetrievalSettings
from app.domain.chat import ChatAnswer
from app.ports.llm import LLMClient
from app.prompts.chat_synthesis import build_single_doc_prompt

logger = logging.getLogger(__name__)


class SingleDocChat:
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        config: RetrievalSettings,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._config = config

    async def answer(self, query: str, doc_id: str) -> ChatAnswer:
        chunks = await self._retriever.retrieve_single(
            query,
            doc_id=doc_id,
            top_k=self._config.single_doc_top_k,
        )

        if not chunks or max(c.score for c in chunks) < self._config.relevance_floor:
            logger.info(
                "Insufficient context for single-doc query (doc_id=%s, top_score=%.3f)",
                doc_id,
                chunks[0].score if chunks else 0.0,
            )
            return ChatAnswer(
                answer="I couldn't find relevant information in the document to answer this question.",
                insufficient_context=True,
            )

        prompt = build_single_doc_prompt(query, chunks)
        logger.debug("Single-doc prompt built (%d chunks, doc_id=%s)", len(chunks), doc_id)

        raw = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            response_schema=ChatAnswer,
        )
        return ChatAnswer.model_validate(raw)
