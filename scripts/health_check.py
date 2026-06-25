"""Quick health-check: verifies Claude, Gemini (LLM + embedder), and Pinecone APIs."""
from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "✓"
FAIL = "✗"


async def check_claude() -> bool:
    from app.adapters.claude_llm import ClaudeClient
    from app.config import get_settings
    try:
        client = ClaudeClient(get_settings().anthropic)
        result = await client.generate(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=16,
        )
        reply = result.get("content", "")
        logger.info("%s Claude — response: %r", PASS, reply)
        return True
    except Exception as exc:
        logger.error("%s Claude — %s", FAIL, exc)
        return False


async def check_gemini_llm() -> bool:
    from app.adapters.gemini_llm import GeminiClient
    from app.config import get_settings
    try:
        client = GeminiClient(get_settings().gemini)
        result = await client.generate(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            # Remove max_tokens=16 or bump it to a safe minimum like 1024
        )
        reply = result.get("content", "")
        logger.info("%s Gemini LLM — response: %r", PASS, reply)
        return True
    except Exception as exc:
        logger.error("%s Gemini LLM — %s", FAIL, exc)
        return False


async def check_gemini_embedder() -> bool:
    from app.adapters.gemini_embedder import GeminiEmbedder
    from app.config import get_settings
    try:
        embedder = GeminiEmbedder(get_settings().gemini)
        vec = await embedder.embed_query("hello world")
        logger.info("%s Gemini Embedder — vector dims: %d", PASS, len(vec))
        return True
    except Exception as exc:
        logger.error("%s Gemini Embedder — %s", FAIL, exc)
        return False


async def check_pinecone() -> bool:
    from pinecone import Pinecone
    from app.config import get_settings
    try:
        settings = get_settings().pinecone
        pc = Pinecone(api_key=settings.api_key.get_secret_value())
        indexes = [i.name for i in pc.list_indexes()]
        logger.info("%s Pinecone — indexes visible: %s", PASS, indexes or "(none yet)")
        return True
    except Exception as exc:
        logger.error("%s Pinecone — %s", FAIL, exc)
        return False


async def main() -> None:
    logger.info("Running health checks …\n")
    results = await asyncio.gather(
        check_claude(),
        check_gemini_llm(),
        check_gemini_embedder(),
        check_pinecone(),
    )
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'='*40}")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
