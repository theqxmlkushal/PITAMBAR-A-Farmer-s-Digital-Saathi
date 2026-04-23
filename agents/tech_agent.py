# agents/tech_agent.py
# Agricultural technology specialist agent.
# Uses market search for equipment prices + RAG for document context.

from __future__ import annotations

import asyncio
import logging

from core.state import PitambarState
from agents.base_agent import run_agent
from tools.apis.market_api import get_mandi_prices
from tools.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

# Common agri-tech equipment terms to search for pricing
_TECH_KEYWORDS = {
    "drip irrigation", "sprayer", "tractor", "pump", "harvester",
    "thresher", "sensor", "greenhouse", "polyhouse", "solar pump",
}


def _extract_equipment(entities: list[str]) -> str:
    """Return the first entity that matches a known equipment keyword, else use first entity."""
    for e in entities:
        if any(k in e.lower() for k in _TECH_KEYWORDS):
            return e
    return entities[0] if entities else "agricultural equipment"


def _build_context(market: dict, rag: str) -> str:
    """Merge equipment price snippets with RAG context."""
    lines: list[str] = []

    results = market.get("results", [])
    if results:
        lines.append("Equipment price and availability data:")
        for i, r in enumerate(results[:4], 1):
            snippet = (r.get("snippet") or "")[:180]
            url     = r.get("url", "")
            lines.append(f"{i}. {snippet}")
            if url:
                lines.append(f"   Source: {url}")

    if rag:
        lines.append("")
        lines.append(rag)

    return "\n".join(lines)


async def run(state: PitambarState) -> dict:
    """Tech agent entry point. Called by the orchestrator."""
    query    = state["query"]
    entities = state["entities"]

    equipment = _extract_equipment(entities)

    # Equipment price search + RAG concurrently
    market_task = asyncio.create_task(get_mandi_prices(equipment, "India"))
    rag_task    = asyncio.create_task(retrieve_context(query))

    market_result = await market_task
    rag_context   = await rag_task

    extra_context = _build_context(market_result, rag_context)

    result = await run_agent(
        agent_name="tech",
        system_prompt_file="tech_prompt.txt",
        query=query,
        entities=entities,
        extra_context=extra_context,
    )

    result["sources"] = [
        r["url"] for r in market_result.get("results", [])[:4] if r.get("url")
    ]
    logger.info("tech_agent done | confidence=%s", result["confidence"])
    return result


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
class TechAgent:
        async def run(self, state: PitambarState) -> dict:
            return await run(state)
if __name__ == "__main__":
    import sys, copy
    from core.state import DEFAULT_STATE

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    async def _test() -> None:
        print("=== tech_agent smoke-test ===\n")
        state: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "query":    "What is the cost of drip irrigation for 1 acre and is there any government subsidy?",
            "entities": ["drip irrigation", "1 acre", "subsidy"],
            "session_id": "test-session",
        }
        result = await run(state)
        print(f"agent      : {result['agent']}")
        print(f"confidence : {result['confidence']}")
        print(f"sources    : {result['sources']}")
        print(f"result     :\n{result['result']}")
        print()
        print("tech_agent OK" if result["confidence"] > 0 else "tech_agent returned error fallback")
        sys.exit(0 if result["confidence"] > 0 else 1)

    asyncio.run(_test())