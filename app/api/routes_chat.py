"""Chat endpoints: POST /chat/single, POST /chat/cross."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.deps import ChatPipelineDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _to_response(answer) -> ChatResponse:
    return ChatResponse(
        answer=answer.answer,
        citations=[c.format() for c in answer.citations],
        insufficient_context=answer.insufficient_context,
    )


@router.post("/single", response_model=ChatResponse)
async def chat_single(body: ChatRequest, pipeline: ChatPipelineDep) -> ChatResponse:
    if not body.doc_id:
        raise HTTPException(status_code=422, detail="doc_id is required for single-doc chat")

    logger.info("Single-doc chat: doc_id=%s query=%r", body.doc_id, body.query[:80])
    answer = await pipeline.answer(body.query, mode="single", doc_id=body.doc_id)
    return _to_response(answer)


@router.post("/cross", response_model=ChatResponse)
async def chat_cross(body: ChatRequest, pipeline: ChatPipelineDep) -> ChatResponse:
    logger.info("Cross-doc chat: query=%r", body.query[:80])
    answer = await pipeline.answer(body.query, mode="cross")
    return _to_response(answer)
