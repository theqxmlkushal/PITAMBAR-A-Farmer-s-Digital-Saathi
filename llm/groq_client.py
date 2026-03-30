# llm/groq_client.py
# Async wrapper around the Groq SDK.
# All agents call `call_llm()` — nothing else in the codebase touches Groq directly.

from __future__ import annotations

import asyncio
import logging
from typing import Final

from groq import AsyncGroq

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level client — instantiated once, reused across all coroutines
# ---------------------------------------------------------------------------
_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Return (or lazily create) the module-level AsyncGroq client."""
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Hard cap on tokens returned by the LLM — prevents runaway responses.
_MAX_TOKENS: Final[int] = 1024


async def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = settings.groq_model,
    temperature: float = 0.3,
) -> str:
    """Send a chat completion request to Groq and return the assistant reply.

    Args:
        system_prompt:  The system-role message (agent persona / instructions).
        user_message:   The user-role message (farmer query + any injected context).
        model:          Groq model ID. Defaults to ``settings.groq_model``.
        temperature:    Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        The assistant's reply as a plain Python string, stripped of leading/trailing
        whitespace.

    Raises:
        RuntimeError: Wraps any Groq SDK or network error with a human-readable message
                      so callers do not need to import Groq exception types.
    """
    logger.debug(
        "call_llm | model=%s | temperature=%s | user_chars=%d",
        model,
        temperature,
        len(user_message),
    )

    try:
        client = _get_client()
        completion = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        content: str = completion.choices[0].message.content or ""
        logger.debug(
            "call_llm | response_chars=%d | finish_reason=%s",
            len(content),
            completion.choices[0].finish_reason,
        )
        return content.strip()

    except Exception as exc:
        logger.exception("Groq API call failed: %s", exc)
        raise RuntimeError(
            f"LLM call failed (model={model}): {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Windows fix: ProactorEventLoop doesn't flush httpx transports cleanly on
    # loop.close().  SelectorEventLoop behaves correctly and is safe on all
    # platforms when we're not doing subprocess I/O (which we aren't).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    _SYSTEM = "You are a helpful assistant. Be concise."
    _USER = "Say hello in one sentence."

    async def _smoke_test() -> None:
        print("=== groq_client smoke-test ===")
        print(f"  Model   : {settings.groq_model}")
        print(
            f"  API key : "
            f"{'[SET]' if settings.groq_api_key != 'your_groq_key_here' else '[PLACEHOLDER — set GROQ_API_KEY in .env]'}"
        )
        print()

        if settings.groq_api_key == "your_groq_key_here":
            print("⚠  GROQ_API_KEY is still a placeholder — skipping live call.")
            sys.exit(0)

        try:
            response = await call_llm(
                system_prompt=_SYSTEM,
                user_message=_USER,
            )
            print(f"Response: {response}")
            print("\ncall_llm ✓")
        except RuntimeError as err:
            print(f"\n✗ call_llm raised RuntimeError:\n  {err}")
            sys.exit(1)
        finally:
            # Explicitly close the shared client so httpx can drain its
            # connection pool before the event loop shuts down.
            # Without this, Windows ProactorEventLoop raises a noisy
            # "Event loop is closed" traceback during garbage collection.
            global _client
            if _client is not None:
                await _client.close()
                _client = None

    asyncio.run(_smoke_test())