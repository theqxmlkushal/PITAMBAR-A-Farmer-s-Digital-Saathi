# core/synthesizer.py
# Final pipeline node: merges one or more agent results into a single,
# coherent, farmer-friendly response.
# Single-agent result → returned directly (no LLM call, saves tokens).
# Multi-agent result  → combined and fed through the synthesis LLM.

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import settings
from core.state import DEFAULT_STATE, PitambarState
from llm.groq_client import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROMPTS_DIR      = Path(__file__).resolve().parents[1] / "llm" / "prompts"
_SYNTHESIS_PROMPT = _PROMPTS_DIR / "synthesis_prompt.txt"

_AGENT_LABELS: dict[str, str] = {
    "market": "Market & Prices",
    "policy": "Government Schemes",
    "tech":   "Farm Technology",
    "crop":   "Crop & Agronomy",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_synthesis_prompt() -> str:
    return _SYNTHESIS_PROMPT.read_text(encoding="utf-8").strip()


def _agent_label(agent_name: str) -> str:
    return _AGENT_LABELS.get(agent_name, agent_name.capitalize())


def _build_combined_input(agent_results: dict) -> str:
    """Format multiple agent outputs into a labeled block for the synthesis LLM."""
    blocks: list[str] = []
    for agent_name, result_dict in agent_results.items():
        result_text = result_dict.get("result", "").strip()
        confidence  = result_dict.get("confidence", 0.0)
        label       = _agent_label(agent_name)

        if not result_text:
            continue

        block = f"[{label}]\n{result_text}"
        if confidence < 0.5:
            block += "\n(Note: this agent reported low confidence — verify independently)"
        blocks.append(block)

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public pipeline node
# ---------------------------------------------------------------------------

async def synthesize(state: PitambarState) -> PitambarState:
    """LangGraph node: merge agent results into state["final_response"].

    Single-agent path: no LLM call — the result is used directly.
    Multi-agent path:  synthesis LLM combines all outputs.
    """
    agent_results: dict = state.get("agent_results", {})

    # Filter to only results that have content
    valid_results = {
        name: r for name, r in agent_results.items()
        if r.get("result", "").strip()
    }

    if not valid_results:
        logger.warning("synthesize() called with no valid agent results")
        return {
            **state,
            "final_response": (
                "I'm sorry, I couldn't find relevant information for your query. "
                "Please try rephrasing or contact your local KVK (helpline: 1800-180-1551)."
            ),
        }

    # --- Single-agent shortcut: skip synthesis LLM ---
    if len(valid_results) == 1:
        agent_name, result_dict = next(iter(valid_results.items()))
        final = result_dict.get("result", "").strip()
        logger.info("synthesize | single-agent='%s' | chars=%d", agent_name, len(final))
        return {**state, "final_response": final}

    # --- Multi-agent: call synthesis LLM ---
    combined = _build_combined_input(valid_results)
    logger.debug(
        "synthesize | agents=%s | combined_chars=%d",
        list(valid_results.keys()), len(combined),
    )

    try:
        system_prompt = _load_synthesis_prompt()
        final_response = await call_llm(
            system_prompt=system_prompt,
            user_message=combined,
            temperature=0.4,   # slightly creative for conversational tone
        )
        logger.info(
            "synthesize | agents=%s | final_chars=%d",
            list(valid_results.keys()), len(final_response),
        )
    except Exception as exc:
        logger.error("synthesize() LLM call failed: %s — falling back to concat", exc)
        # Graceful degradation: concatenate all results with headers
        final_response = "\n\n".join(
            f"{_agent_label(n)}:\n{r['result']}"
            for n, r in valid_results.items()
        )

    return {**state, "final_response": final_response}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio, copy, sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    # --- Mock agent results ---
    _MOCK_SINGLE: dict = {
        "crop": {
            "agent":      "crop",
            "result":     "Wheat rust disease shows orange pustules on leaves. Apply propiconazole 25% EC at 1ml/litre. Avoid overhead irrigation during infection period.",
            "sources":    ["https://agrifarming.in/wheat-rust"],
            "confidence": 0.9,
        }
    }

    _MOCK_MULTI: dict = {
        "crop": {
            "agent":      "crop",
            "result":     "Wheat rust disease shows orange pustules on leaves. Apply propiconazole 25% EC at 1ml/litre. Avoid overhead irrigation during infection period.",
            "sources":    ["https://agrifarming.in/wheat-rust"],
            "confidence": 0.9,
        },
        "policy": {
            "agent":      "policy",
            "result":     "PMFBY (Pradhan Mantri Fasal Bima Yojana) covers wheat crop losses due to disease. Premium is only 1.5% for Rabi crops. Register at your nearest bank or at pmfby.gov.in before the cutoff date.",
            "sources":    ["https://pmfby.gov.in"],
            "confidence": 0.9,
        },
    }

    async def _test() -> None:
        print("=== synthesizer smoke-test ===\n")

        # Test 1: Single agent — should return result directly, no LLM call
        print("Test 1 — single agent (no LLM call expected)")
        state_single: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "query":        "wheat rust treatment",
            "agent_results": _MOCK_SINGLE,
            "session_id":   "test-single",
        }
        result_single = await synthesize(state_single)
        print(f"final_response ({len(result_single['final_response'])} chars):")
        print(f"  {result_single['final_response']}")
        print()

        # Test 2: Multi-agent — should call synthesis LLM
        print("Test 2 — multi-agent (synthesis LLM call expected)")
        state_multi: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "query":        "wheat rust disease, am I eligible for PMFBY?",
            "agent_results": _MOCK_MULTI,
            "session_id":   "test-multi",
        }
        result_multi = await synthesize(state_multi)
        print(f"final_response ({len(result_multi['final_response'])} chars):")
        print(result_multi["final_response"])
        print()

        # Test 3: Empty agent results — graceful fallback
        print("Test 3 — empty agent results (fallback message expected)")
        state_empty: PitambarState = {
            **copy.deepcopy(DEFAULT_STATE),
            "agent_results": {},
            "session_id":   "test-empty",
        }
        result_empty = await synthesize(state_empty)
        print(f"Fallback: {result_empty['final_response']}")
        print()

        all_ok = (
            result_single["final_response"] == _MOCK_SINGLE["crop"]["result"]
            and len(result_multi["final_response"]) > 50
            and "KVK" in result_empty["final_response"]
        )
        print("synthesizer ✓" if all_ok else "⚠  one or more tests produced unexpected output")
        sys.exit(0 if all_ok else 1)

    asyncio.run(_test())