"""
memory/session_memory.py
------------------------
In-memory session store for PITAMBAR / FasalMitra.
No external DB — pure Python dict, TTL-based expiry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Config (mirrors settings.py so this file stays self-contained for testing)
# ---------------------------------------------------------------------------
try:
    from config.settings import SESSION_TTL_SECONDS  # type: ignore
except ImportError:
    SESSION_TTL_SECONDS = 3600  # 1 hour default


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMemory:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    last_active: str = field(default_factory=_utcnow_iso)


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_STORE: dict[str, SessionMemory] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create(session_id: str) -> SessionMemory:
    """Return existing session or create a fresh one."""
    if session_id not in _STORE:
        _STORE[session_id] = SessionMemory(session_id=session_id)
    return _STORE[session_id]


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Append a message to the session.

    Parameters
    ----------
    session_id : str
    role       : "user" | "assistant"
    content    : message text
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")

    session = get_or_create(session_id)
    session.messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": _utcnow_iso(),
        }
    )
    session.last_active = _utcnow_iso()


def get_history(session_id: str, last_n: int = 6) -> list[dict]:
    """
    Return the last *last_n* messages for the session.
    Returns an empty list if the session does not exist.
    """
    session = _STORE.get(session_id)
    if session is None:
        return []
    return session.messages[-last_n:]


def clear_session(session_id: str) -> None:
    """Delete a session entirely from the store."""
    _STORE.pop(session_id, None)


def cleanup_expired() -> int:
    """
    Remove all sessions whose *last_active* timestamp is older than
    SESSION_TTL_SECONDS.  Returns the number of sessions removed.
    """
    now = datetime.now(timezone.utc)
    expired = [
        sid
        for sid, sess in _STORE.items()
        if (now - datetime.fromisoformat(sess.last_active)).total_seconds()
        > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _STORE[sid]
    return len(expired)


# ---------------------------------------------------------------------------
# __main__ smoke-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("session_memory.py — smoke tests")
    print("=" * 60)

    SID = "test-session-001"

    # 1. get_or_create — new session
    sess = get_or_create(SID)
    assert sess.session_id == SID
    assert sess.messages == []
    print(f"[PASS] get_or_create  → session_id={sess.session_id}")

    # 2. get_or_create — idempotent
    sess2 = get_or_create(SID)
    assert sess is sess2
    print("[PASS] get_or_create is idempotent")

    # 3. add_message
    add_message(SID, "user", "Meri fasal ko paani ki zaroorat hai?")
    add_message(SID, "assistant", "Wheat needs ~450 mm of water per season.")
    add_message(SID, "user", "How often should I irrigate?")
    add_message(SID, "assistant", "Irrigate every 10–12 days at tillering stage.")
    add_message(SID, "user", "What about fertilizer?")
    add_message(SID, "assistant", "Apply 120 kg/ha urea in split doses.")
    add_message(SID, "user", "Thanks!")
    assert len(_STORE[SID].messages) == 7
    print(f"[PASS] add_message    → {len(_STORE[SID].messages)} messages stored")

    # 4. get_history — default last_n=6
    hist = get_history(SID)
    assert len(hist) == 6
    # 7 messages total; last 6 starts at index 1 (assistant turn)
    assert hist[0]["role"] == "assistant"
    assert hist[0]["content"] == "Wheat needs ~450 mm of water per season."
    print(f"[PASS] get_history(last_n=6) → {len(hist)} messages")

    # 5. get_history — custom last_n
    hist2 = get_history(SID, last_n=2)
    assert len(hist2) == 2
    assert hist2[-1]["content"] == "Thanks!"
    print(f"[PASS] get_history(last_n=2) → {len(hist2)} messages")

    # 6. get_history — non-existent session
    assert get_history("ghost-session") == []
    print("[PASS] get_history on missing session → []")

    # 7. invalid role
    try:
        add_message(SID, "system", "bad role")
        print("[FAIL] Should have raised ValueError")
    except ValueError as exc:
        print(f"[PASS] ValueError on bad role → {exc}")

    # 8. cleanup_expired — nothing should be removed yet
    removed = cleanup_expired()
    assert removed == 0
    print(f"[PASS] cleanup_expired (fresh sessions) → removed={removed}")

    # 9. cleanup_expired — simulate expired session
    import importlib, sys

    SID2 = "old-session-999"
    get_or_create(SID2)
    # Manually backdate last_active
    _STORE[SID2].last_active = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    removed = cleanup_expired()
    assert removed == 1
    assert SID2 not in _STORE
    print(f"[PASS] cleanup_expired (expired session) → removed={removed}")

    # 10. clear_session
    clear_session(SID)
    assert SID not in _STORE
    print(f"[PASS] clear_session  → session removed")

    print("\nAll session_memory tests passed ✓")