"""
core/orchestrator.py
LangGraph StateGraph pipeline for PITAMBAR multi-agent agricultural advisory system.
Wires together routing, RAG retrieval, agent execution, synthesis, and memory.
"""

import asyncio
import logging
from dataclasses import asdict
from typing import Literal

from langgraph.graph import StateGraph, START, END

from core.state import PitambarState, DEFAULT_STATE
from core import router, synthesizer
import agents.market_agent as market_agent
import agents.policy_agent as policy_agent
import agents.tech_agent   as tech_agent
import agents.crop_agent   as crop_agent
import memory.session_memory as session_memory
import memory.user_profile  as user_profile
from tools.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent registry — maps intent name -> agent module's run() coroutine
# ---------------------------------------------------------------------------

class _AgentWrapper:
    """Thin wrapper so AGENT_MAP values have a uniform .run(state) interface."""
    def __init__(self, run_fn):
        self._run = run_fn

    async def run(self, state: PitambarState) -> dict:
        return await self._run(state)


AGENT_MAP: dict[str, _AgentWrapper] = {
    "market": _AgentWrapper(market_agent.run),
    "policy": _AgentWrapper(policy_agent.run),
    "tech":   _AgentWrapper(tech_agent.run),
    "crop":   _AgentWrapper(crop_agent.run),
}

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def enrich_state_node(state: PitambarState) -> PitambarState:
    """
    Loads conversation history and user profile into state.
    Extracts lightweight profile hints from the current query
    (e.g. crops, location mentions) and persists them.
    """
    sid = state["session_id"]

    # Load conversation history
    history = session_memory.get_history(sid)
    state["conversation_history"] = history

    # Load or create profile, extract hints, persist
    profile = user_profile.get_profile(sid)
    user_profile.extract_profile_hints(state["query"], profile)   # mutates + persists in-place
    state["user_profile"] = asdict(profile)                        # store as plain dict in state

    logger.debug("[enrich_state_node] session=%s profile=%s", sid, state["user_profile"])
    return state


async def rag_node(state: PitambarState) -> PitambarState:
    context = await retrieve_context(state["query"])   # ✅ properly awaited
    state["rag_context"] = context
    logger.debug("[rag_node] retrieved %d chars of context", len(context or ""))
    return state


async def router_node(state: PitambarState) -> PitambarState:
    """
    Classifies the query into one of:
    market | policy | tech | crop | multi
    and resolves which agents are needed.
    """
    updated = await router.classify(state)
    logger.debug(
        "[router_node] intent=%s agents_needed=%s",
        updated.get("intent"),
        updated.get("agents_needed"),
    )
    return updated


def route_decision(state: PitambarState) -> Literal["single_agent", "parallel_agents"]:
    """
    Conditional edge function (not a node).
    Directs flow to single-agent or parallel-agents branch
    depending on how many agents the router selected.
    """
    agents = state.get("agents_needed", [])
    return "single_agent" if len(agents) == 1 else "parallel_agents"


async def single_agent_node(state: PitambarState) -> PitambarState:
    """
    Runs the single required agent and writes its result into
    state["agent_results"] keyed by agent name.
    """
    agent_name = state["agents_needed"][0]
    agent      = AGENT_MAP[agent_name]
    result     = await agent.run(state)
    state["agent_results"] = {agent_name: result}
    logger.debug(
        "[single_agent_node] agent=%s confidence=%s",
        agent_name, result.get("confidence"),
    )
    return state


async def parallel_agents_node(state: PitambarState) -> PitambarState:
    """
    Runs all required agents concurrently via asyncio.gather and
    merges every result into state["agent_results"].
    """
    agents_needed = state["agents_needed"]
    tasks         = [AGENT_MAP[name].run(state) for name in agents_needed]
    results       = await asyncio.gather(*tasks, return_exceptions=False)
    state["agent_results"] = {
        name: result for name, result in zip(agents_needed, results)
    }
    logger.debug("[parallel_agents_node] ran agents=%s", agents_needed)
    return state


async def synthesizer_node(state: PitambarState) -> PitambarState:
    """
    Combines all agent results into a single coherent final_response
    using the synthesis prompt.
    """
    updated = await synthesizer.synthesize(state)
    logger.debug(
        "[synthesizer_node] final_response length=%d",
        len(updated.get("final_response", "")),
    )
    return updated


def save_memory_node(state: PitambarState) -> PitambarState:
    """
    Persists the user query and assistant response to session memory
    so multi-turn context is available on the next request.
    """
    sid = state["session_id"]
    session_memory.add_message(sid, role="user",      content=state["query"])
    session_memory.add_message(sid, role="assistant", content=state.get("final_response", ""))
    logger.debug("[save_memory_node] saved turn to session=%s", sid)
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(PitambarState)

    g.add_node("enrich_state",    enrich_state_node)
    g.add_node("rag",             rag_node)
    g.add_node("router",          router_node)
    g.add_node("single_agent",    single_agent_node)
    g.add_node("parallel_agents", parallel_agents_node)
    g.add_node("synthesizer",     synthesizer_node)
    g.add_node("save_memory",     save_memory_node)

    # Linear spine
    g.add_edge(START,          "enrich_state")
    g.add_edge("enrich_state", "rag")
    g.add_edge("rag",          "router")

    # Conditional branch after routing
    g.add_conditional_edges(
        "router",
        route_decision,
        {
            "single_agent":    "single_agent",
            "parallel_agents": "parallel_agents",
        },
    )

    # Both branches converge at synthesizer
    g.add_edge("single_agent",    "synthesizer")
    g.add_edge("parallel_agents", "synthesizer")

    g.add_edge("synthesizer", "save_memory")
    g.add_edge("save_memory", END)

    return g


pipeline = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

async def process(query: str, session_id: str = "default") -> dict:
    """
    Builds a fresh PitambarState, runs the compiled pipeline,
    and returns a clean response dict.

    Args:
        query:      The farmer's natural-language question.
        session_id: Identifier for session-scoped memory / profile.

    Returns:
        {
            "query":          str,
            "intent":         str,
            "final_response": str,
            "agents_used":    list[str],
            "sources":        list[str],
            "session_id":     str,
        }
    """
    import copy
    initial_state: PitambarState = {
        **copy.deepcopy(DEFAULT_STATE),
        "query":      query,
        "session_id": session_id,
    }

    final_state: PitambarState = await pipeline.ainvoke(initial_state)

    # Collect sources from every agent result
    all_sources: list[str] = []
    for agent_result in final_state.get("agent_results", {}).values():
        all_sources.extend(agent_result.get("sources", []))

    return {
        "query":          final_state["query"],
        "intent":         final_state.get("intent", "unknown"),
        "final_response": final_state.get("final_response", ""),
        "agents_used":    list(final_state.get("agent_results", {}).keys()),
        "sources":        list(dict.fromkeys(all_sources)),   # deduplicated, order-stable
        "session_id":     final_state["session_id"],
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # ← NEW: insert project root so `core`, `agents`, `memory` etc. are importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    TEST_QUERY = (
        "What crop should I grow in Vidarbha in June "
        "and is there any government support?"
    )

    async def _smoke_test():
        logging.basicConfig(level=logging.DEBUG)
        print("=" * 60)
        print("PITAMBAR orchestrator smoke-test")
        print("=" * 60)
        print(f"Query : {TEST_QUERY}\n")
        response = await process(TEST_QUERY, session_id="smoke_test_001")
        print(f"Intent         : {response['intent']}")
        print(f"Agents used    : {response['agents_used']}")
        print(f"Sources        : {response['sources']}")
        print(f"\nFinal response :\n{response['final_response']}")
        print("=" * 60)

    asyncio.run(_smoke_test())
    sys.exit(0)