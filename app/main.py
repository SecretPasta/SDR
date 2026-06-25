"""FastAPI application factory."""
from __future__ import annotations

import logging
import logging.config

from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_compare import router as compare_router
from app.deps import get_gemini_embedder, get_pinecone_store, get_app_settings

_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "app": {"level": "DEBUG", "propagate": True},
        "httpx": {"level": "WARNING", "propagate": True},
    },
}


def create_app() -> FastAPI:
    logging.config.dictConfig(_LOGGING)
    logger = logging.getLogger(__name__)

    app = FastAPI(
        title="FDS Reconciler",
        description="AI-powered comparison of Functional Design Specification documents.",
        version="0.1.0",
    )

    app.include_router(compare_router)
    app.include_router(chat_router)

    @app.on_event("startup")
    async def _warmup() -> None:
        """Prime the Pinecone connection with a no-op query so the first real request is fast."""
        try:
            settings = get_app_settings()
            store    = get_pinecone_store(settings)
            embedder = get_gemini_embedder(settings)
            dummy    = await embedder.embed_query("warmup")
            await store.query(dummy, top_k=1)
            logger.info("Pinecone warmup complete")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pinecone warmup failed (non-fatal): %s", exc)

    return app


app = create_app()
