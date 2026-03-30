# core/state.py
# Shared state schema for the PITAMBAR LangGraph pipeline.
# This file is pure schema — no logic, no imports from other PITAMBAR modules.
# Every node in the StateGraph reads from and writes to a PitambarState dict.

from __future__ import annotations

from typing import Optional
try:
    from typing import TypedDict               # Python 3.11+
except ImportError:
    from typing_extensions import TypedDict    # fallback for 3.8–3.10


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PitambarState(TypedDict):
    # --- Input ---
    query: str
    """Raw query string from the farmer (text or transcribed voice)."""

    session_id: str
    """Unique session identifier. Used to look up conversation history."""

    user_profile: dict
    """Farmer profile dict: {name, location, crops, language, ...}
    Populated by memory/user_profile.py before the pipeline starts."""

    # --- Routing ---
    intent: str
    """Classified intent: 'market' | 'policy' | 'tech' | 'crop' | 'multi' | ''
    Empty string means not yet classified."""

    entities: list[str]
    """Keywords extracted from the query by the router (crop names, locations, etc.)."""

    agents_needed: list[str]
    """Which agent(s) the orchestrator should invoke for this query.
    Single-intent: ['crop']  |  Multi-intent: ['market', 'policy']"""

    # --- Processing ---
    rag_context: str
    """Relevant document context retrieved from the vector store.
    Injected into each agent's user message before the LLM call."""

    agent_results: dict
    """Keyed by agent name. Each value:
    {
        'agent':      str,
        'result':     str,
        'sources':    list[str],
        'confidence': float,
    }"""

    # --- Memory ---
    conversation_history: list[dict]
    """Ordered list of prior turns for this session:
    [{'role': 'user'|'assistant', 'content': str}, ...]"""

    # --- Output ---
    final_response: str
    """Synthesized natural-language response ready to send to the farmer."""

    audio_output: Optional[bytes]
    """TTS-encoded audio bytes. None unless the request came in via /query/voice."""

    # --- Error handling ---
    error: Optional[str]
    """Non-None if any pipeline node encountered an unrecoverable error.
    Downstream nodes should short-circuit when this field is set."""


# ---------------------------------------------------------------------------
# Default state — use as a starting template for every new request
# ---------------------------------------------------------------------------

DEFAULT_STATE: PitambarState = {
    "query":                "",
    "session_id":           "",
    "user_profile":         {},
    "intent":               "",
    "entities":             [],
    "agents_needed":        [],
    "rag_context":          "",
    "agent_results":        {},
    "conversation_history": [],
    "final_response":       "",
    "audio_output":         None,
    "error":                None,
}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import copy

    print("=== core/state.py smoke-test ===\n")

    # 1. Verify all expected keys are present
    expected_keys = {
        "query", "session_id", "user_profile",
        "intent", "entities", "agents_needed",
        "rag_context", "agent_results", "conversation_history",
        "final_response", "audio_output", "error",
    }
    actual_keys = set(DEFAULT_STATE.keys())
    missing = expected_keys - actual_keys
    extra   = actual_keys - expected_keys

    print("Expected keys :", sorted(expected_keys))
    print("Actual keys   :", sorted(actual_keys))
    assert not missing, f"Missing keys: {missing}"
    assert not extra,   f"Unexpected keys: {extra}"
    print("Key check      : ✓\n")

    # 2. Verify default types
    assert isinstance(DEFAULT_STATE["query"],                str),  "query must be str"
    assert isinstance(DEFAULT_STATE["entities"],             list), "entities must be list"
    assert isinstance(DEFAULT_STATE["agents_needed"],        list), "agents_needed must be list"
    assert isinstance(DEFAULT_STATE["agent_results"],        dict), "agent_results must be dict"
    assert isinstance(DEFAULT_STATE["conversation_history"], list), "conversation_history must be list"
    assert isinstance(DEFAULT_STATE["user_profile"],         dict), "user_profile must be dict"
    assert DEFAULT_STATE["audio_output"] is None,                   "audio_output default must be None"
    assert DEFAULT_STATE["error"]        is None,                   "error default must be None"
    print("Type checks    : ✓\n")

    # 3. Verify DEFAULT_STATE is not mutated across copies
    state_a = copy.deepcopy(DEFAULT_STATE)
    state_b = copy.deepcopy(DEFAULT_STATE)
    state_a["entities"].append("wheat")
    assert "wheat" not in state_b["entities"], "deepcopy isolation failed"
    print("Isolation check: ✓\n")

    # 4. Show a sample filled state
    sample: PitambarState = {
        **DEFAULT_STATE,
        "query":      "What is the mandi price of wheat in Punjab today?",
        "session_id": "sess-abc123",
        "intent":     "market",
        "entities":   ["wheat", "Punjab", "mandi price"],
        "agents_needed": ["market"],
        "user_profile":  {"name": "Gurpreet", "location": "Ludhiana", "language": "pa"},
    }
    print("Sample state fields:")
    for k, v in sample.items():
        print(f"  {k:<24} : {v!r}")

    print("\ncore/state.py ✓")