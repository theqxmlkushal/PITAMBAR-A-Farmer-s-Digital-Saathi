# agents/market_agent.py
# Market specialist agent.
# Fetches live mandi prices and formats them as LLM context.

from __future__ import annotations

import asyncio
import logging

from core.state import PitambarState
from agents.base_agent import run_agent
from tools.apis.market_api import get_mandi_prices

logger = logging.getLogger(__name__)


def _build_context(results: list[dict]) -> str:
    """Format DDG mandi snippets into a numbered context block."""
    if not results:
        return ""
    lines = ["Live mandi price data:"]
    for i, r in enumerate(results[:5], 1):
        snippet = (r.get("snippet") or "")[:200]
        url     = r.get("url", "")
        lines.append(f"{i}. {snippet}")
        if url:
            lines.append(f"   Source: {url}")
    return "\n".join(lines)


async def run(state: PitambarState) -> dict:
    """Market agent entry point. Called by the orchestrator."""
    query    = state["query"]
    entities = state["entities"]

    # Find crop and optional location from entities
    skip = {"today", "india", "farmer", "price", "mandi", "market", "rate"}
    crops    = [e for e in entities if e.lower() not in skip]
    location = ""
    for e in entities:
        if e.istitle() and e.lower() not in skip:
            location = e
            break

    crop = crops[0] if crops else query.split()[0]

    market_result = await get_mandi_prices(crop, location)
    extra_context = _build_context(market_result.get("results", []))

    result = await run_agent(
        agent_name="market",
        system_prompt_file="market_prompt.txt",
        query=query,
        entities=entities,
        extra_context=extra_context,
    )

    result["sources"] = [
        r["url"] for r in market_result.get("results", [])[:5] if r.get("url")
    ]
    logger.info("market_agent done | confidence=%s", result["confidence"])
    return result


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
class MarketAgent:
    async def run(self, state: PitambarState) -> dict:
        return await run(state)
if __name__ == "__main__":
    import sys, copy
    from core.state import DEFAULT_STATE

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    async def _test() -> None:
        print("=== market_agent smoke-test ===\n")
        state: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "query":    "What is the current mandi price of onion in Nashik?",
            "entities": ["onion", "Nashik"],
            "session_id": "test-session",
        }
        result = await run(state)
        print(f"agent      : {result['agent']}")
        print(f"confidence : {result['confidence']}")
        print(f"sources    : {result['sources']}")
        print(f"result     :\n{result['result']}")
        print()
        print("market_agent OK" if result["confidence"] > 0 else "market_agent returned error fallback")
        sys.exit(0 if result["confidence"] > 0 else 1)

    asyncio.run(_test())