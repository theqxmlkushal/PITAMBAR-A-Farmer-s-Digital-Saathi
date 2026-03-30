# tools/apis/govt_scheme_api.py
# Searches for Indian government agricultural schemes via DuckDuckGo text search.
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
_MAX_RESULTS   = 5
_QUERY_SUFFIX  = "government scheme India farmer subsidy"


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

async def search_schemes(query: str) -> dict[str, Any]:
    """Search for government agricultural schemes related to *query*.

    The query is automatically enriched with scheme-specific keywords so
    callers can pass a raw farmer question (e.g. "crop insurance Maharashtra").

    Returns a dict with keys:
        results (list[dict])— up to 5 dicts with title / url / snippet
        error   (str|None)
    """
    enriched = f"{query.strip()} {_QUERY_SUFFIX}"
    logger.debug("search_schemes | query='%s'", enriched)

    try:
        results = await asyncio.to_thread(
            _ddg_text_search, enriched, _MAX_RESULTS
        )
        logger.debug("search_schemes | hits=%d", len(results))
        return {"results": results, "error": None}

    except Exception as exc:
        logger.warning("search_schemes failed for '%s': %s", query, exc)
        return {"results": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test() -> None:
        print("=== govt_scheme_api smoke-test ===\n")

        queries = [
            "PM-KISAN eligibility",
            "PMFBY crop insurance wheat",
            "drip irrigation subsidy Maharashtra",
        ]

        for q in queries:
            result = await search_schemes(q)
            print(f"Query : {q}")
            if result["error"]:
                print(f"Error : {result['error']}")
            else:
                for i, r in enumerate(result["results"], 1):
                    print(f"  {i}. {r['title']}")
                    print(f"     {r['url']}")
                    print(f"     {r['snippet'][:120]}...")
            print()

    asyncio.run(_test())