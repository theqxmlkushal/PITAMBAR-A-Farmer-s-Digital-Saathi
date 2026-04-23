"""
PITAMBAR — multimodal/speech_to_text.py
Speech-to-text using openai-whisper (local, free).
Model is loaded lazily on first call and cached module-level.

REQUIREMENT: ffmpeg must be installed and available on PATH.
  Windows : winget install ffmpeg   OR   choco install ffmpeg
  Linux   : sudo apt install ffmpeg
  macOS   : brew install ffmpeg
"""

import asyncio
import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy cache
# ---------------------------------------------------------------------------
_whisper_model = None


# ---------------------------------------------------------------------------
# ffmpeg pre-flight check
# ---------------------------------------------------------------------------
def _check_ffmpeg() -> None:
    """Raise RuntimeError with a clear install message if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — Whisper requires it to decode audio.\n"
            "  Windows : winget install ffmpeg   OR  choco install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg\n"
            "  macOS   : brew install ffmpeg\n"
            "After installing, restart your terminal so PATH is refreshed."
        )


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
def _get_model():
    """Load Whisper model once and cache it; also verifies ffmpeg is present."""
    global _whisper_model
    if _whisper_model is None:
        _check_ffmpeg()                          # fail fast with a clear message
        try:
            import whisper                       # noqa: PLC0415
            try:
                from config.settings import settings  # noqa: PLC0415
                model_name: str = getattr(settings, "WHISPER_MODEL", "base")
            except Exception:
                model_name = "base"

            logger.info(
                "Loading Whisper model '%s' (first call — may download ~150 MB)…",
                model_name,
            )
            _whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model '%s' loaded successfully.", model_name)
        except Exception as exc:
            logger.error("Failed to load Whisper model: %s", exc)
            raise
    return _whisper_model


# ---------------------------------------------------------------------------
# Core sync helper — runs in asyncio.to_thread
# ---------------------------------------------------------------------------
def _transcribe_sync(audio_bytes: bytes, language: str) -> dict:
    tmp_path: str | None = None
    try:
        model = _get_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        # File is now CLOSED — safe for ffmpeg / Whisper to open on Windows

        result = model.transcribe(tmp_path, language=language)

        text: str = result.get("text", "").strip()
        detected_lang: str = result.get("language", language)

        segments = result.get("segments", [])
        if segments:
            avg_lp = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            confidence = float(max(0.0, min(1.0, 1.0 + avg_lp)))
        else:
            confidence = 0.0

        return {"text": text, "language": detected_lang, "confidence": confidence, "error": None}

    except Exception as exc:
        logger.error("Whisper transcription error: %s", exc)
        return {"text": "", "language": language, "confidence": 0.0, "error": str(exc)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning("Could not delete temp file %s: %s", tmp_path, e)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------
async def transcribe(audio_bytes: bytes, language: str = "hi") -> dict:
    """
    Transcribe raw audio bytes using Whisper.

    Args:
        audio_bytes: Raw WAV/MP3/etc. content (anything ffmpeg can decode).
        language:    BCP-47 hint, e.g. "hi", "en", "mr".

    Returns:
        {"text": str, "language": str, "confidence": float, "error": str|None}
    """
    if not audio_bytes:
        return {
            "text": "", "language": language,
            "confidence": 0.0, "error": "Empty audio bytes received.",
        }
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, language)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    async def _test():
        print("=== PITAMBAR: speech_to_text self-test ===\n")

        # Check ffmpeg before anything else so we fail with a clear message
        try:
            _check_ffmpeg()
            print("ffmpeg : found ✓")
        except RuntimeError as e:
            print(f"ffmpeg : NOT FOUND ✗\n\n{e}")
            sys.exit(1)

        if len(sys.argv) > 1:
            wav_path = sys.argv[1]
            print(f"Audio file : {wav_path}")
            with open(wav_path, "rb") as f:
                audio = f.read()
        else:
            print("No audio file supplied — pass a WAV/MP3 path as argument:")
            print("  python -m multimodal.speech_to_text path/to/audio.wav\n")
            print("Generating a 1-second silent WAV for smoke-test...")
            import struct
            sample_rate, num_samples = 16000, 16000   # 1 second silence
            data_size = num_samples * 2
            audio = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", 36 + data_size, b"WAVE",
                b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
                b"data", data_size,
            ) + bytes(data_size)

        result = await transcribe(audio, language="hi")
        print("\n--- Result ---")
        if result["error"]:
            print(f"Error      : {result['error']}")
        else:
            print(f"Text       : {result['text']!r}")
            print(f"Language   : {result['language']}")
            print(f"Confidence : {result['confidence']:.3f}")

    asyncio.run(_test())