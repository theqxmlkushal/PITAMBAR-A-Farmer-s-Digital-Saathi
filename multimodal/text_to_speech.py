"""
PITAMBAR — multimodal/text_to_speech.py
Text-to-speech synthesis using gTTS (free, no API key).
Returns raw MP3 bytes suitable for streaming or file writing.
"""

import asyncio
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language code mapping
# gTTS uses BCP-47-style codes; keep a clean normalisation table here so
# callers can pass either our internal codes or raw BCP-47 tags.
# ---------------------------------------------------------------------------
_LANGUAGE_MAP: dict[str, str] = {
    "hi": "hi",   # Hindi
    "en": "en",   # English
    "mr": "mr",   # Marathi
    "pa": "pa",   # Punjabi
    "gu": "gu",   # Gujarati
    "bn": "bn",   # Bengali
    "te": "te",   # Telugu
    "ta": "ta",   # Tamil
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
}

_FALLBACK_LANGUAGE = "hi"


def _resolve_language(language: str) -> str:
    """Return the gTTS-compatible language code, falling back to Hindi."""
    code = _LANGUAGE_MAP.get(language.lower().split("-")[0])
    if code is None:
        logger.warning(
            "Language '%s' not in supported map — falling back to '%s'.",
            language,
            _FALLBACK_LANGUAGE,
        )
        return _FALLBACK_LANGUAGE
    return code


# ---------------------------------------------------------------------------
# Core sync helper — runs in a thread via asyncio.to_thread
# ---------------------------------------------------------------------------
def _synthesize_sync(text: str, language: str) -> bytes:
    """
    Use gTTS to convert text → MP3 bytes.
    Intentionally sync; called via asyncio.to_thread.
    """
    try:
        from gtts import gTTS  # noqa: PLC0415

        gtts_lang = _resolve_language(language)
        tts = gTTS(text=text, lang=gtts_lang, slow=False)

        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()

    except Exception as exc:
        logger.error("gTTS synthesis error: %s", exc)
        return b""


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------
async def synthesize_speech(text: str, language: str = "hi") -> bytes:
    """
    Convert text to speech and return raw MP3 bytes.

    Args:
        text:     The text to synthesize.
        language: BCP-47 language code (e.g. "hi", "en", "mr").
                  Unknown codes fall back to Hindi.

    Returns:
        Raw MP3 bytes, or empty bytes on failure.
    """
    if not text or not text.strip():
        logger.warning("synthesize_speech called with empty text — returning empty bytes.")
        return b""

    return await asyncio.to_thread(_synthesize_sync, text, language)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    async def _test():
        print("=== PITAMBAR: text_to_speech self-test ===")

        test_cases = [
            ("नमस्ते किसान भाइयों, आज का मौसम अच्छा है।", "hi"),
            ("Hello farmers, today's weather looks good.", "en"),
            ("नमस्कार शेतकरी बंधूंनो, आजचे हवामान चांगले आहे.", "mr"),
        ]

        for text, lang in test_cases:
            print(f"\nLanguage : {lang}")
            print(f"Text     : {text}")
            mp3_bytes = await synthesize_speech(text, language=lang)
            if mp3_bytes:
                print(f"MP3 size : {len(mp3_bytes):,} bytes  ✓")

                # Optionally write to disk for manual playback verification.
                if "--save" in sys.argv:
                    out_path = f"test_tts_{lang}.mp3"
                    with open(out_path, "wb") as f:
                        f.write(mp3_bytes)
                    print(f"Saved    : {out_path}")
            else:
                print("ERROR    : synthesis returned empty bytes.")

        # Edge-case: empty text
        print("\n--- Edge case: empty string ---")
        result = await synthesize_speech("", "hi")
        print(f"Empty text → {len(result)} bytes (expected 0)  {'✓' if result == b'' else '✗'}")

        # Edge-case: unsupported language code (should fallback gracefully)
        print("\n--- Edge case: unsupported language code 'xx' ---")
        result = await synthesize_speech("Fallback test.", "xx")
        print(f"Unknown lang → {len(result):,} bytes  {'✓' if result else '✗'}")

    asyncio.run(_test())