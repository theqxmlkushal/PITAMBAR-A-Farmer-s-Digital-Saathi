# agents/crop_agent.py
# Agronomy specialist agent.
# Enriches queries with live weather data and RAG context before calling the LLM.

from __future__ import annotations

import asyncio
import logging

from core.state import PitambarState
from agents.base_agent import run_agent
from tools.apis.weather_api import get_forecast
from tools.apis.market_api import get_mandi_prices
from tools.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_weather(forecast: dict) -> str:
    """Convert a weather forecast dict into a compact context string."""
    if forecast.get("error") or not forecast.get("days"):
        return ""
    loc = forecast["location"]
    today = forecast["days"][0]
    return (
        f"Weather in {loc}: "
        f"max {today['max_temp']}°C, min {today['min_temp']}°C, "
        f"rain {today['rain_mm']}mm, wind {today['wind_kmh']} km/h."
    )


def _format_market(market: dict) -> str:
    """Pull the top snippet from a mandi price result."""
    snippets = [r["snippet"] for r in market.get("results", [])[:2] if r.get("snippet")]
    if not snippets:
        return ""
    return "Market context: " + " | ".join(s[:120] for s in snippets)


def _find_location(entities: list[str]) -> str:
    """Heuristic: return the first entity that looks like an Indian city/state."""
    indian_places = {
        "punjab", "maharashtra", "karnataka", "gujarat", "rajasthan",
        "up", "mp", "bihar", "haryana", "andhra", "telangana", "kerala",
        "pune", "delhi", "mumbai", "nagpur", "ludhiana", "jaipur",
        "hyderabad", "bengaluru", "chennai", "kolkata", "patna", "lucknow",
    }
    for e in entities:
        if e.lower() in indian_places:
            return e
    return entities[0] if entities else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run(state: PitambarState) -> dict:
    """Crop agent entry point. Called by the orchestrator."""
    query    = state["query"]
    entities = state["entities"]

    location = _find_location(entities)

    # Fire weather + RAG concurrently; market lookup for any crop entity
    crop_entity = next(
        (e for e in entities if e.lower() not in {"today", "india", "farmer"}),
        query.split()[0],
    )

    weather_task = asyncio.create_task(get_forecast(location)) if location else None
    market_task  = asyncio.create_task(get_mandi_prices(crop_entity, location))
    rag_task     = asyncio.create_task(retrieve_context(query))

    weather_result = await weather_task if weather_task else {"days": [], "error": "no location"}
    market_result  = await market_task
    rag_context    = await rag_task

    # Build extra_context block
    parts: list[str] = []
    weather_str = _format_weather(weather_result)
    if weather_str:
        parts.append(weather_str)
    market_str = _format_market(market_result)
    if market_str:
        parts.append(market_str)
    if rag_context:
        parts.append(rag_context)

    extra_context = "\n".join(parts)

    result = await run_agent(
        agent_name="crop",
        system_prompt_file="crop_prompt.txt",
        query=query,
        entities=entities,
        extra_context=extra_context,
    )

    # Attach any source URLs from market results
    result["sources"] = [r["url"] for r in market_result.get("results", [])[:3] if r.get("url")]
    logger.info("crop_agent done | confidence=%s", result["confidence"])
    return result


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
class CropAgent:
    async def run(self, state: PitambarState) -> dict:
        return await run(state)
    
if __name__ == "__main__":
    import sys
    from core.state import DEFAULT_STATE
    import copy

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    async def _test() -> None:
        print("=== crop_agent smoke-test ===\n")
        state: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "query":    "My tomato leaves are turning yellow and curling. What disease is this and how do I treat it?",
            "entities": ["tomato", "Pune"],
            "session_id": "test-session",
        }
        result = await run(state)
        print(f"agent      : {result['agent']}")
        print(f"confidence : {result['confidence']}")
        print(f"sources    : {result['sources']}")
        print(f"result     :\n{result['result']}")
        print()
        print("crop_agent ✓" if result["confidence"] > 0 else "✗ returned error fallback")
        sys.exit(0 if result["confidence"] > 0 else 1)

    asyncio.run(_test())