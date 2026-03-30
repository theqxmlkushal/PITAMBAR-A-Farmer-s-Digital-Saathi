# tools/scraping/rss_parser.py
# Fetches agricultural news from RSS feeds concurrently.
# feedparser is synchronous — each feed is wrapped in asyncio.to_thread.
# Never raises — returns an empty list on any error.

from __future__ import annotations

import asyncio
import logging
from typing import Any

import feedparser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed registry — add or remove URLs here only, nothing else needs changing.
# User-Agent is sent with every request so feeds don't block the default
# Python/feedparser UA string.
# ---------------------------------------------------------------------------
RSS_FEEDS: list[str] = [
    "https://www.downtoearth.org.in/rss/agriculture",
    "https://krishijagran.com/feed/",
    "https://agrifarming.in/feed",
    "https://www.thehindubusinessline.com/economy/agri-business/feeder/default.rss",
]

_USER_AGENT = "Mozilla/5.0 (compatible; PITAMBAR-AgriBot/1.0)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_feed(url: str) -> list[dict[str, str]]:
    """Blocking feedparser call — must be invoked via asyncio.to_thread."""
    try:
        parsed = feedparser.parse(
            url,
            request_headers={"User-Agent": _USER_AGENT},
        )

        # feedparser never raises — a failed fetch sets bozo=True
        if parsed.bozo and not parsed.entries:
            logger.warning(
                "_parse_feed | url=%s | bozo=True | exc=%s",
                url,
                parsed.get("bozo_exception", "unknown"),
            )
            return []

        items: list[dict[str, str]] = []
        for entry in parsed.entries:
            raw_summary: str = entry.get("summary", "") or ""
            clean_summary = raw_summary.replace("<p>", "").replace("</p>", "").strip()
            items.append(
                {
                    "title":     entry.get("title", "").strip(),
                    "summary":   clean_summary,
                    "url":       entry.get("link", "").strip(),
                    "published": entry.get("published", "").strip(),
                }
            )

        logger.debug("_parse_feed | url=%s | entries=%d", url, len(items))
        return items

    except Exception as exc:
        logger.warning("_parse_feed failed for '%s': %s", url, exc)
        return []


def _matches(item: dict[str, str], keyword: str) -> bool:
    """Return True if keyword appears in item title or summary (case-insensitive)."""
    kw = keyword.lower()
    return kw in item["title"].lower() or kw in item["summary"].lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_agri_news(
    keyword: str = "",
    max_items: int = 8,
) -> list[dict[str, str]]:
    """Fetch agricultural news from all registered RSS feeds concurrently."""
    tasks = [asyncio.to_thread(_parse_feed, url) for url in RSS_FEEDS]
    results: list[Any] = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[dict[str, str]] = []
    for outcome in results:
        if isinstance(outcome, Exception):
            logger.warning("fetch_agri_news: a feed task raised %s", outcome)
            continue
        all_items.extend(outcome)

    kw = keyword.strip()
    if kw:
        all_items = [item for item in all_items if _matches(item, kw)]

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in all_items:
        url = item["url"]
        if url and url not in seen:
            seen.add(url)
            deduped.append(item)

    logger.debug(
        "fetch_agri_news | keyword='%s' | total=%d | deduped=%d | returning=%d",
        kw, len(all_items), len(deduped), min(len(deduped), max_items),
    )
    return deduped[:max_items]


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print(f"feedparser version : {feedparser.__version__}")
    print(f"feeds registered   : {len(RSS_FEEDS)}")
    print()

    print("--- Per-feed diagnostic (sync) ---")
    for feed_url in RSS_FEEDS:
        d = feedparser.parse(feed_url, request_headers={"User-Agent": _USER_AGENT})
        status  = d.get("status", "N/A")
        entries = len(d.entries)
        bozo    = d.bozo
        bozo_ex = str(d.get("bozo_exception", "")) if bozo else ""
        flag    = "OK" if entries > 0 else ("BOZO" if bozo else "EMPTY")
        print(f"  [{flag:5}] status={status} entries={entries:3}  {feed_url}")
        if bozo_ex:
            print(f"           bozo_exc: {bozo_ex}")
    print()

    async def _test() -> None:
        print("--- Async fetch tests ---\n")

        print("Test 1 — keyword='wheat'")
        items = await fetch_agri_news(keyword="wheat", max_items=8)
        if items:
            for i, item in enumerate(items, 1):
                print(f"  {i}. {item['title']}")
                print(f"     {item['published']}  {item['url']}")
        else:
            print("  (no results — 'wheat' not in current headlines)")
        print()

        print("Test 2 — no keyword (all articles, max 5)")
        all_items = await fetch_agri_news(keyword="", max_items=5)
        for i, item in enumerate(all_items, 1):
            print(f"  {i}. {item['title']}")
        if not all_items:
            print("  (no items — check per-feed diagnostic above)")
        print()

        print("Test 3 — keyword='xyznonexistent'")
        none_items = await fetch_agri_news(keyword="xyznonexistent")
        print(f"  Results: {len(none_items)} (expected 0)")
        print()

        ok = len(all_items) > 0
        print("rss_parser OK" if ok else "WARNING: all feeds returned 0 entries")
        sys.exit(0 if ok else 1)

    asyncio.run(_test())