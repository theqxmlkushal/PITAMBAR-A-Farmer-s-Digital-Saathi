# core/router.py
# Intent classification node for the PITAMBAR LangGraph pipeline.
# Classifies the farmer's query into one of: market | policy | tech | crop | multi
# and extracts entities and required agents.

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config.settings import settings
from core.state import DEFAULT_STATE, PitambarState
from llm.groq_client import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROMPTS_DIR   = Path(__file__).resolve().parents[1] / "llm" / "prompts"
_ROUTER_PROMPT = _PROMPTS_DIR / "router_prompt.txt"

_ALL_AGENTS    = ["market", "policy", "tech", "crop"]
_VALID_INTENTS = {"market", "policy", "tech", "crop", "multi"}

_FALLBACK_ROUTING: dict = {
    "intent":        "crop",
    "entities":      [],
    "agents_needed": ["crop"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_router_prompt() -> str:
    return _ROUTER_PROMPT.read_text(encoding="utf-8").strip()


def _strip_markdown(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences and stray backticks."""
    text = text.strip()
    # Remove fenced code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_router_response(raw: str) -> dict:
    """Parse the LLM output into a routing dict.

    Strips markdown, locates the JSON object, and falls back gracefully
    on any parse error.
    """
    cleaned = _strip_markdown(raw)

    # Attempt to extract the first {...} block in case there's surrounding text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Router JSON parse failed: %s | raw=%r", exc, raw[:200])
        return dict(_FALLBACK_ROUTING)

    # Validate and normalise intent
    intent: str = str(parsed.get("intent", "")).lower().strip()
    if intent not in _VALID_INTENTS:
        logger.warning("Router returned unknown intent '%s' — falling back to crop", intent)
        return dict(_FALLBACK_ROUTING)

    entities: list[str] = [
        str(e).strip() for e in parsed.get("entities", []) if str(e).strip()
    ]

    agents_raw: list[str] = [
        str(a).strip().lower() for a in parsed.get("agents_needed", [])
    ]
    agents_needed: list[str] = [a for a in agents_raw if a in _ALL_AGENTS]

    # Enforce invariants
    if intent == "multi" and not agents_needed:
        logger.debug("multi-intent but agents_needed empty — using all four agents")
        agents_needed = list(_ALL_AGENTS)

    if intent != "multi":
        # Single-intent: agents_needed must be exactly [intent]
        agents_needed = [intent]

    return {
        "intent":        intent,
        "entities":      entities,
        "agents_needed": agents_needed,
    }


# ---------------------------------------------------------------------------
# Public pipeline node
# ---------------------------------------------------------------------------

async def classify(state: PitambarState) -> PitambarState:
    """LangGraph node: classify intent and extract entities from the query.

    Reads state["query"], calls the router LLM, and returns the state
    with intent / entities / agents_needed populated.
    """
    query = state.get("query", "").strip()
    if not query:
        logger.warning("classify() called with empty query — applying fallback")
        return {
            **state,
            **_FALLBACK_ROUTING,
            "error": "Empty query received",
        }

    try:
        system_prompt = _load_router_prompt()
        raw_response  = await call_llm(
            system_prompt=system_prompt,
            user_message=query,
            temperature=0.0,   # deterministic — routing must be consistent
        )
        logger.debug("Router raw response: %r", raw_response[:300])

        routing = _parse_router_response(raw_response)

    except Exception as exc:
        logger.error("classify() LLM call failed: %s — using fallback", exc)
        routing = dict(_FALLBACK_ROUTING)

    logger.info(
        "classify | intent=%s | agents=%s | entities=%s",
        routing["intent"], routing["agents_needed"], routing["entities"],
    )
    return {**state, **routing}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio, copy, sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    _TEST_QUERIES = [
        "onion mandi price in Nashik today",
        "PM-KISAN eligibility",
        "my wheat has rust disease and I want drip irrigation subsidy",
        "best tractor for 5 acres",
        "kharif crop advice for Vidarbha",
    ]

    _EXPECTED_INTENTS = ["market", "policy", "multi", "tech", "crop"]

    async def _test() -> None:
        print("=== router smoke-test ===\n")
        all_passed = True

        for i, query in enumerate(_TEST_QUERIES):
            state: PitambarState = {
                **copy.deepcopy(DEFAULT_STATE),
                "query": query,
                "session_id": f"test-{i}",
            }
            result = await classify(state)
            intent  = result["intent"]
            agents  = result["agents_needed"]
            entities = result["entities"]
            expected = _EXPECTED_INTENTS[i]
            match = "✓" if intent == expected else f"? (expected {expected})"

            print(f"Query    : {query}")
            print(f"Intent   : {intent}  {match}")
            print(f"Agents   : {agents}")
            print(f"Entities : {entities}")
            print()

            if intent not in _VALID_INTENTS:
                all_passed = False

        print("router ✓" if all_passed else "⚠  one or more intents were unexpected")
        sys.exit(0 if all_passed else 1)

    asyncio.run(_test())