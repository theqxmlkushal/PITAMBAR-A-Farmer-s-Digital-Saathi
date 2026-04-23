"""
PITAMBAR — app/controllers/query_controller.py
Bridges HTTP layer → orchestrator pipeline → multimodal output.
"""

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation guard
# ---------------------------------------------------------------------------
# The LangGraph state dict contains ``audio_output: bytes | None``.
# Raw bytes are NOT JSON-serialisable — FastAPI will either raise a 500 or
# emit garbled binary to the client.  _sanitize() walks the result dict and:
#   • converts bytes  → base64 string  (keeps the data, makes it safe)
#   • converts sets   → sorted list    (sets are not JSON-serialisable)
#   • drops None values at the top level so the response stays lean
# ---------------------------------------------------------------------------
def _sanitize(data: dict) -> dict:
    """Return a JSON-safe copy of *data*, recursively converting problem types."""

    def _clean(value: Any) -> Any:
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("utf-8") if value else ""
        if isinstance(value, set):
            return sorted(_clean(v) for v in value)
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        return value

    return {k: _clean(v) for k, v in data.items() if v is not None or k in ("error", "final_response")}


# ---------------------------------------------------------------------------
# Text query
# ---------------------------------------------------------------------------
async def handle_text_query(message: str, session_id: str) -> dict:
    """
    Run the full PITAMBAR pipeline for a plain-text query.

    Args:
        message:    Farmer's question (any supported language).
        session_id: Caller-supplied session identifier.

    Returns:
        Full state dict produced by the orchestrator, guaranteed to contain
        at least ``final_response`` and ``error`` keys.
    """
    from core.orchestrator import process  # noqa: PLC0415 — avoid circular at import time

    try:
        result: dict = await process(message, session_id)
        return _sanitize(result)
    except Exception as exc:
        logger.exception("handle_text_query failed for session '%s': %s", session_id, exc)
        return {"final_response": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Voice query
# ---------------------------------------------------------------------------
async def handle_voice_query(
    audio_bytes: bytes,
    session_id: str,
    response_language: str = "hi",
) -> dict:
    """
    Full multimodal pipeline: audio → text → orchestrator → TTS → response.

    Args:
        audio_bytes:       Raw WAV/audio bytes from the caller.
        session_id:        Caller-supplied session identifier.
        response_language: BCP-47 code used for both STT hint and TTS output.

    Returns:
        Orchestrator result dict extended with ``audio_base64`` (MP3 encoded
        as base64 string).  On transcription failure returns
        ``{"error": "Could not transcribe audio"}``.
    """
    from multimodal.speech_to_text import transcribe          # noqa: PLC0415
    from multimodal.text_to_speech import synthesize_speech   # noqa: PLC0415
    from core.orchestrator import process                      # noqa: PLC0415

    # ── 1. Speech → Text ────────────────────────────────────────────────────
    stt_result = await transcribe(audio_bytes, language=response_language)

    if stt_result.get("error") or not stt_result.get("text", "").strip():
        logger.warning(
            "STT failed for session '%s': %s", session_id, stt_result.get("error")
        )
        return {"error": "Could not transcribe audio"}

    transcribed_text: str = stt_result["text"]
    logger.info("STT [%s] → %r", session_id, transcribed_text[:120])

    # ── 2. Text → Orchestrator ───────────────────────────────────────────────
    try:
        result: dict = await process(transcribed_text, session_id)
    except Exception as exc:
        logger.exception("Orchestrator failed for session '%s': %s", session_id, exc)
        return {"error": str(exc)}

    # ── 3. Final response → Speech ───────────────────────────────────────────
    final_text: str = result.get("final_response", "")
    audio_mp3: bytes = b""

    if final_text.strip():
        try:
            audio_mp3 = await synthesize_speech(final_text, language=response_language)
        except Exception as exc:
            logger.error("TTS failed for session '%s': %s", session_id, exc)
            # Non-fatal — return text response even if TTS fails

    audio_b64: str = base64.b64encode(audio_mp3).decode("utf-8") if audio_mp3 else ""

    # Sanitize BEFORE merging audio_base64 so the orchestrator's raw
    # ``audio_output: bytes`` field (if present) is also cleaned up.
    clean = _sanitize(result)
    clean["audio_base64"] = audio_b64          # already a str — safe
    clean.pop("audio_output", None)            # redundant now; audio_base64 replaces it
    return clean


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _test():
        print("=== PITAMBAR: query_controller self-test ===")

        # -- _sanitize unit test ---------------------------------------------
        print("\n[0] _sanitize() correctness check")
        raw = {
            "final_response": "अच्छी फसल के लिए सिंचाई जरूरी है।",
            "audio_output": b"\xff\xfb\x90",        # raw bytes — must become b64 str
            "agent_results": {"market": {"sources": {"a", "b"}}},  # set → list
            "error": None,                           # None kept for mandatory keys
            "session_id": None,                      # None on non-mandatory → dropped
        }
        clean = _sanitize(raw)
        assert isinstance(clean["audio_output"], str), "bytes not converted!"
        assert isinstance(clean["agent_results"]["market"]["sources"], list), "set not converted!"
        assert "session_id" not in clean, "None non-mandatory key not dropped!"
        assert clean["error"] is None, "mandatory None key should be kept!"
        print("  _sanitize ✓")

        # -- Text query smoke test -------------------------------------------
        print("\n[1] Text query")
        res = await handle_text_query(
            message="What is the MSP for wheat this year?",
            session_id="test-session-001",
        )
        print("Keys returned :", list(res.keys()))
        print("final_response:", str(res.get("final_response", ""))[:200])
        print("error         :", res.get("error"))
        # Verify no bytes in top-level values
        for k, v in res.items():
            assert not isinstance(v, bytes), f"Bytes leaked in key '{k}'!"
        print("  No raw bytes in response ✓")

        # -- Voice query with empty audio (exercises error path) ---------------
        print("\n[2] Voice query — empty audio (expect error)")
        res2 = await handle_voice_query(b"", "test-session-002", "hi")
        print("Result:", res2)
        assert res2.get("error") == "Could not transcribe audio"
        print("  Error path ✓")

    asyncio.run(_test())