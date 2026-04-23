# agents/policy_agent.py
# Government scheme specialist agent.
# Searches official scheme data via DuckDuckGo + RAG context.

from __future__ import annotations

import asyncio
import logging

from core.state import PitambarState
from agents.base_agent import run_agent
from tools.apis.govt_scheme_api import search_schemes
from tools.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)


def _build_context(results: list[dict], rag: str) -> str:
    """Merge DDG scheme results with RAG context into one block."""
    lines: list[str] = []

    if results:
        lines.append("Government scheme search results:")
        for i, r in enumerate(results[:5], 1):
            snippet = (r.get("snippet") or "")[:200]
            url     = r.get("url", "")
            lines.append(f"{i}. {snippet}")
            if url:
                lines.append(f"   Source: {url}")

    if rag:
        lines.append("")
        lines.append(rag)

    return "\n".join(lines)


async def run(state: PitambarState) -> dict:
    """Policy agent entry point. Called by the orchestrator."""
    query    = state["query"]
    entities = state["entities"]

    # Run scheme search and RAG concurrently
    scheme_task = asyncio.create_task(search_schemes(query))
    rag_task    = asyncio.create_task(retrieve_context(query))

    scheme_result = await scheme_task
    rag_context   = await rag_task

    extra_context = _build_context(scheme_result.get("results", []), rag_context)

    result = await run_agent(
        agent_name="policy",
        system_prompt_file="policy_prompt.txt",
        query=query,
        entities=entities,
        extra_context=extra_context,
    )

    result["sources"] = [
        r["url"] for r in scheme_result.get("results", [])[:5] if r.get("url")
    ]
    logger.info("policy_agent done | confidence=%s", result["confidence"])
    return result


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
class PolicyAgent:
    async def run(self, state: PitambarState) -> dict:
        return await run(state)

if __name__ == "__main__":
    import sys, copy
    from core.state import DEFAULT_STATE


    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    async def _test() -> None:
        print("=== policy_agent smoke-test ===\n")
        state: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "query":    "Am I eligible for PM-KISAN? I have 2 acres of land in Maharashtra.",
            "entities": ["PM-KISAN", "Maharashtra", "2 acres"],
            "session_id": "test-session",
        }
        result = await run(state)
        print(f"agent      : {result['agent']}")
        print(f"confidence : {result['confidence']}")
        print(f"sources    : {result['sources']}")
        print(f"result     :\n{result['result']}")
        print()
        print("policy_agent OK" if result["confidence"] > 0 else "policy_agent returned error fallback")
        sys.exit(0 if result["confidence"] > 0 else 1)

    asyncio.run(_test())