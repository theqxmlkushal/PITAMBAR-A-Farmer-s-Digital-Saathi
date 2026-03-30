# tools/apis/market_api.py
# Fetches mandi/commodity price information via DuckDuckGo text search.
# Runs DDGS in a thread so the async event loop is never blocked.
# No API key required. Never raises — all errors are returned in the dict.

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ddgs import DDGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RESULTS = 5


# ---------------------------------------------------------------------------
# Internal helper — runs in a thread pool via asyncio.to_thread
# ---------------------------------------------------------------------------

def _ddg_text_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Blocking DuckDuckGo text search. Must be called via asyncio.to_thread."""
    with DDGS() as ddgs:
        raw = ddgs.text(query, max_results=max_results)
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in (raw or [])
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_mandi_prices(crop: str, location: str = "") -> dict[str, Any]:
    """Search for current mandi prices for *crop*, optionally near *location*.

    Returns a dict with keys:
        crop    (str)       — the queried crop name
        results (list[dict])— up to 5 dicts with title / url / snippet
        error   (str|None)
    """
    location_part = location.strip()
    # "Rs per quintal" + "agmarknet" forces Indian agri-market results and
    # avoids the crop name matching unrelated content (router firmware, Wikipedia, etc.)
    query = (
        f"{crop} mandi price Rs per quintal {location_part} India agmarknet 2026"
    ).strip()
    logger.debug("get_mandi_prices | query='%s'", query)

    try:
        results = await asyncio.to_thread(
            _ddg_text_search, query, _MAX_RESULTS
        )
        logger.debug("get_mandi_prices | hits=%d", len(results))
        return {"crop": crop, "results": results, "error": None}

    except Exception as exc:
        logger.warning("get_mandi_prices failed for '%s': %s", crop, exc)
        return {"crop": crop, "results": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test() -> None:
        print("=== market_api smoke-test ===\n")

        test_cases = [
            ("tomato", "Pune"),
            ("wheat",  "Punjab"),
            ("onion",  ""),
        ]

        for crop, loc in test_cases:
            result = await get_mandi_prices(crop, loc)
            label = f"{crop}" + (f" / {loc}" if loc else "")
            print(f"Query : {label}")
            if result["error"]:
                print(f"Error : {result['error']}")
            else:
                for i, r in enumerate(result["results"], 1):
                    print(f"  {i}. {r['title']}")
                    print(f"     {r['url']}")
                    print(f"     {r['snippet'][:120]}...")
            print()

    asyncio.run(_test())