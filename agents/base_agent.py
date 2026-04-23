# agents/base_agent.py
# Shared execution core for all PITAMBAR agents.
# Every specialist agent calls run_agent() — no agent talks to Groq directly.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from llm.groq_client import call_llm

logger = logging.getLogger(__name__)

# Resolve prompts directory relative to this file so it works regardless of cwd
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "llm" / "prompts"


# ---------------------------------------------------------------------------
# Prompt loader (sync — tiny file read, no need for to_thread)
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    """Read and return the contents of llm/prompts/{filename}."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent(
    agent_name: str,
    system_prompt_file: str,
    query: str,
    entities: list[str],
    extra_context: str = "",
) -> dict:
    """Execute one agent turn: load prompt → build message → call LLM → return result.

    Args:
        agent_name:         Logical name ("crop", "market", etc.) — used in return dict.
        system_prompt_file: Filename inside llm/prompts/ (e.g. "crop_prompt.txt").
        query:              The farmer's original query string.
        entities:           Keyword list extracted by the router.
        extra_context:      Additional context block (weather, RAG, search snippets).

    Returns:
        {"agent": str, "result": str, "sources": list, "confidence": float}
    """
    try:
        system_prompt = _load_prompt(system_prompt_file)

        entities_str = ", ".join(entities) if entities else "none"
        user_message = (
            f"Query: {query}\n"
            f"Entities: {entities_str}\n"
            f"Context:\n{extra_context or 'No additional context available.'}"
        )

        logger.debug(
            "run_agent | agent=%s | query_chars=%d | context_chars=%d",
            agent_name, len(query), len(extra_context),
        )

        result = await call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
        )

        return {
            "agent":      agent_name,
            "result":     result,
            "sources":    [],
            "confidence": 0.9,
        }

    except Exception as exc:
        logger.exception("run_agent failed | agent=%s | error=%s", agent_name, exc)
        return {
            "agent":      agent_name,
            "result":     "I could not process this query at the moment. Please try again.",
            "sources":    [],
            "confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    async def _test() -> None:
        print("=== base_agent smoke-test ===\n")

        result = await run_agent(
            agent_name="crop",
            system_prompt_file="crop_prompt.txt",
            query="My wheat leaves are turning yellow. What should I do?",
            entities=["wheat", "yellow leaves"],
            extra_context="Weather: max 32°C, min 18°C, no rain expected.",
        )

        print(f"agent      : {result['agent']}")
        print(f"confidence : {result['confidence']}")
        print(f"sources    : {result['sources']}")
        print(f"result     :\n{result['result']}")
        print()
        print("base_agent ✓" if result["confidence"] > 0 else "✗ base_agent returned error fallback")
        sys.exit(0 if result["confidence"] > 0 else 1)

    asyncio.run(_test())