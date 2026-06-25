from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from app.chat.retriever import RetrievedChunk, Retriever
from app.config import RetrievalSettings
from app.domain.chat import ChatAnswer, Citation
from app.ports.embedder import EmbedderClient
from app.ports.llm import LLMClient
from app.prompts.chat_synthesis import build_cross_doc_prompt, build_single_doc_prompt

logger = logging.getLogger(__name__)

_FALLBACK_TOP_K = 6


# ── flat LLM output schema ────────────────────────────────────────────────────
# ChatAnswer contains a nested Citation model with page: int | None, which
# causes Gemini's structured-output JSON generation to emit truncated responses.
# This flat schema has no nested models and no union types.

class _SynthesisOutput(BaseModel):
    answer: str
    citations: list[str] = []  # formatted strings: "filename · §section · page N"
    insufficient_context: bool = False


def _parse_citation(s: str) -> Citation:
    """Parse a formatted citation string back into a Citation object."""
    parts = [p.strip() for p in s.split(" · ")]
    filename = parts[0] if parts else s
    section = parts[1].lstrip("§") if len(parts) > 1 else ""
    page: int | None = None
    if len(parts) > 2 and parts[2].startswith("page "):
        with contextlib.suppress(ValueError):
            page = int(parts[2].removeprefix("page "))
    return Citation(filename=filename, section=section, page=page)


# ── state ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    query: str
    mode: Literal["single", "cross"]
    doc_id: str | None
    query_embedding: list[float]
    context_chunks: dict[str, list[RetrievedChunk]]
    sufficiency_ok: bool
    answer: ChatAnswer


# ── helpers ───────────────────────────────────────────────────────────────────

def _max_score(chunks: list[RetrievedChunk]) -> float:
    return max((c.score for c in chunks), default=0.0)


def _all_chunks(chunks_by_doc: dict[str, list[RetrievedChunk]]) -> list[RetrievedChunk]:
    return [c for chunks in chunks_by_doc.values() for c in chunks]


# ── pipeline ──────────────────────────────────────────────────────────────────

class ChatPipeline:
    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbedderClient,
        retriever: Retriever,
        config: RetrievalSettings,
        doc_ids: tuple[str, str] = ("A", "B"),
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._retriever = retriever
        self._config = config
        self._doc_ids = doc_ids
        self._graph = self._build()

    # ── nodes ─────────────────────────────────────────────────────────────────

    async def _embed_query(self, state: ChatState) -> dict[str, Any]:
        vector = await self._embedder.embed_query(state["query"])
        logger.debug("Embedded query (%d dims)", len(vector))
        return {"query_embedding": vector}

    async def _retrieve_single(self, state: ChatState) -> dict[str, Any]:
        doc_id = state["doc_id"] or self._doc_ids[0]
        chunks = await self._retriever.retrieve_single(
            state["query"],
            doc_id=doc_id,
            top_k=self._config.single_doc_top_k,
        )
        logger.debug("retrieve_single: %d chunks (doc_id=%s)", len(chunks), doc_id)
        return {"context_chunks": {doc_id: chunks}}

    async def _retrieve_cross(self, state: ChatState) -> dict[str, Any]:
        chunks_by_doc = await self._retriever.retrieve_dual(
            state["query"],
            doc_ids=self._doc_ids,
            top_k_per_doc=self._config.cross_doc_top_k,
        )

        floor = self._config.relevance_floor
        both_below = all(
            _max_score(chunks_by_doc.get(did, [])) < floor for did in self._doc_ids
        )

        if both_below:
            logger.info(
                "Both sides below relevance floor %.2f — trying unfiltered fallback", floor
            )
            fallback = await self._retriever.retrieve_unfiltered(
                state["query"], top_k=_FALLBACK_TOP_K
            )
            grouped: dict[str, list[RetrievedChunk]] = {}
            for chunk in fallback:
                did = chunk.metadata.get("doc_id", "unknown")
                grouped.setdefault(did, []).append(chunk)
            chunks_by_doc = grouped

        logger.debug(
            "retrieve_cross: %s",
            {did: len(v) for did, v in chunks_by_doc.items()},
        )
        return {"context_chunks": chunks_by_doc}

    def _check_sufficiency(self, state: ChatState) -> dict[str, Any]:
        best = _max_score(_all_chunks(state["context_chunks"]))
        ok = best >= self._config.relevance_floor
        logger.debug("check_sufficiency: best_score=%.3f, ok=%s", best, ok)
        return {"sufficiency_ok": ok}

    async def _synthesize(self, state: ChatState) -> dict[str, Any]:
        chunks_by_doc = state["context_chunks"]
        if state["mode"] == "single":
            doc_id = state["doc_id"] or self._doc_ids[0]
            chunks = chunks_by_doc.get(doc_id, [])
            prompt = build_single_doc_prompt(state["query"], chunks)
        else:
            prompt = build_cross_doc_prompt(state["query"], chunks_by_doc)

        raw = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            response_schema=_SynthesisOutput,
        )
        output = _SynthesisOutput.model_validate(raw)
        return {"answer": ChatAnswer(
            answer=output.answer,
            citations=[_parse_citation(c) for c in output.citations],
            insufficient_context=output.insufficient_context,
        )}

    @staticmethod
    def _insufficient_response(_state: ChatState) -> dict[str, Any]:
        return {
            "answer": ChatAnswer(
                answer="I couldn't find relevant information to answer this question.",
                insufficient_context=True,
            )
        }

    # ── routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _route_mode(state: ChatState) -> str:
        return "retrieve_single" if state["mode"] == "single" else "retrieve_cross"

    @staticmethod
    def _route_after_check(state: ChatState) -> str:
        return "synthesize" if state["sufficiency_ok"] else "insufficient_response"

    # ── graph construction ────────────────────────────────────────────────────

    def _build(self) -> Any:
        g: StateGraph = StateGraph(ChatState)  # type: ignore[arg-type]

        g.add_node("embed_query",          self._embed_query)          # type: ignore[arg-type]
        g.add_node("retrieve_single",      self._retrieve_single)      # type: ignore[arg-type]
        g.add_node("retrieve_cross",       self._retrieve_cross)       # type: ignore[arg-type]
        g.add_node("check_sufficiency",    self._check_sufficiency)    # type: ignore[arg-type]
        g.add_node("synthesize",           self._synthesize)           # type: ignore[arg-type]
        g.add_node("insufficient_response", self._insufficient_response)  # type: ignore[arg-type]

        g.add_edge(START, "embed_query")
        g.add_conditional_edges("embed_query", self._route_mode)
        g.add_edge("retrieve_single",  "check_sufficiency")
        g.add_edge("retrieve_cross",   "check_sufficiency")
        g.add_conditional_edges("check_sufficiency", self._route_after_check)
        g.add_edge("synthesize",           END)
        g.add_edge("insufficient_response", END)

        return g.compile()

    # ── entry point ───────────────────────────────────────────────────────────

    async def answer(
        self,
        query: str,
        mode: Literal["single", "cross"],
        doc_id: str | None = None,
    ) -> ChatAnswer:
        if mode == "single" and doc_id is None:
            raise ValueError("doc_id is required for mode='single'")

        initial: ChatState = {
            "query": query,
            "mode": mode,
            "doc_id": doc_id,
            "query_embedding": [],
            "context_chunks": {},
            "sufficiency_ok": False,
            "answer": ChatAnswer(answer=""),
        }
        final = await self._graph.ainvoke(initial)
        return final["answer"]
