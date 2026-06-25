"""Dependency injection wiring — the only place where concrete types are instantiated."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.adapters.claude_llm import ClaudeClient
from app.adapters.gemini_embedder import GeminiEmbedder
from app.adapters.gemini_llm import GeminiClient
from app.adapters.pinecone_store import PineconeStore
from app.chat.graph import ChatPipeline
from app.chat.retriever import Retriever
from app.comparison.aligner import HeadingAligner
from app.comparison.explainer import MissingExplainer
from app.comparison.judge import PairwiseJudge
from app.comparison.pipeline import ComparisonPipeline
from app.comparison.ranker import Top10Ranker
from app.config import AppSettings, get_settings


# ── settings ──────────────────────────────────────────────────────────────────

def get_app_settings() -> AppSettings:
    return get_settings()


SettingsDep = Annotated[AppSettings, Depends(get_app_settings)]


# ── adapter singletons ────────────────────────────────────────────────────────

@lru_cache
def _claude_client(settings: AppSettings = Depends(get_app_settings)) -> ClaudeClient:
    return ClaudeClient(settings.anthropic)


@lru_cache
def _gemini_client(settings: AppSettings = Depends(get_app_settings)) -> GeminiClient:
    return GeminiClient(settings.gemini)


@lru_cache
def _gemini_embedder(settings: AppSettings = Depends(get_app_settings)) -> GeminiEmbedder:
    return GeminiEmbedder(settings.gemini)


@lru_cache
def _pinecone_store(settings: AppSettings = Depends(get_app_settings)) -> PineconeStore:
    return PineconeStore(settings.pinecone, dimension=settings.gemini.embed_dimensions)


# ── FastAPI-compatible providers (not cached via lru_cache — FastAPI manages scope) ──

def get_claude_client(settings: SettingsDep) -> ClaudeClient:
    return _claude_client(settings)


def get_gemini_client(settings: SettingsDep) -> GeminiClient:
    return _gemini_client(settings)


def get_gemini_embedder(settings: SettingsDep) -> GeminiEmbedder:
    return _gemini_embedder(settings)


def get_pinecone_store(settings: SettingsDep) -> PineconeStore:
    return _pinecone_store(settings)


ClaudeClientDep  = Annotated[ClaudeClient,  Depends(get_claude_client)]
GeminiClientDep  = Annotated[GeminiClient,  Depends(get_gemini_client)]
GeminiEmbedderDep = Annotated[GeminiEmbedder, Depends(get_gemini_embedder)]
PineconeStoreDep = Annotated[PineconeStore,  Depends(get_pinecone_store)]


# ── pipeline providers ────────────────────────────────────────────────────────

def get_comparison_pipeline(
    settings: SettingsDep,
    llm: ClaudeClientDep,
    embedder: GeminiEmbedderDep,
) -> ComparisonPipeline:
    aligner   = HeadingAligner(embedder, settings.alignment)
    judge     = PairwiseJudge(llm)
    explainer = MissingExplainer(llm)
    ranker    = Top10Ranker(llm)
    return ComparisonPipeline(
        llm=llm,
        embedder=embedder,
        aligner=aligner,
        judge=judge,
        explainer=explainer,
        ranker=ranker,
    )


def get_chat_pipeline(
    settings: SettingsDep,
    llm: GeminiClientDep,
    embedder: GeminiEmbedderDep,
    store: PineconeStoreDep,
) -> ChatPipeline:
    retriever = Retriever(embedder, store)
    return ChatPipeline(
        llm=llm,
        embedder=embedder,
        retriever=retriever,
        config=settings.retrieval,
    )


ComparisonPipelineDep = Annotated[ComparisonPipeline, Depends(get_comparison_pipeline)]
ChatPipelineDep       = Annotated[ChatPipeline,       Depends(get_chat_pipeline)]
