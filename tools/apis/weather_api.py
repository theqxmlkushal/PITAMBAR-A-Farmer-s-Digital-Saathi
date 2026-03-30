# tools/apis/weather_api.py
# Fetches a 5-day weather forecast for any Indian location using Open-Meteo.
# No API key required. Never raises — all errors are returned in the dict.

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL  = settings.open_meteo_base_url
_TIMEOUT       = httpx.Timeout(10.0)
_DAILY_PARAMS  = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _geocode(client: httpx.AsyncClient, location: str) -> tuple[float, float, str]:
    """Return (latitude, longitude, resolved_name) for the given location string.

    Raises:
        ValueError: when the geocoding API returns no results.
    """
    resp = await client.get(
        _GEOCODING_URL,
        params={"name": location, "count": 1},
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    results = data.get("results")
    if not results:
        raise ValueError(f"Location not found: '{location}'")

    hit = results[0]
    resolved = f"{hit.get('name', location)}, {hit.get('country', '')}"
    return float(hit["latitude"]), float(hit["longitude"]), resolved.strip(", ")


async def _fetch_forecast(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
) -> dict[str, Any]:
    """Return raw Open-Meteo forecast JSON."""
    resp = await client.get(
        _FORECAST_URL,
        params={
            "latitude":      lat,
            "longitude":     lon,
            "daily":         _DAILY_PARAMS,
            "forecast_days": 5,
            "timezone":      "Asia/Kolkata",
        },
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_forecast(location: str) -> dict[str, Any]:
    """Fetch a 5-day weather forecast for *location*.

    Returns a dict with keys:
        location (str)   — resolved place name
        days     (list)  — list of daily dicts (date, max_temp, min_temp,
                           rain_mm, wind_kmh)
        error    (str|None)
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            lat, lon, resolved = await _geocode(client, location)
            raw = await _fetch_forecast(client, lat, lon)

        daily = raw.get("daily", {})
        dates     = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        rain      = daily.get("precipitation_sum", [])
        wind      = daily.get("windspeed_10m_max", [])

        days = [
            {
                "date":     dates[i],
                "max_temp": max_temps[i],
                "min_temp": min_temps[i],
                "rain_mm":  rain[i],
                "wind_kmh": wind[i],
            }
            for i in range(len(dates))
        ]

        logger.debug("get_forecast | location=%s | days=%d", resolved, len(days))
        return {"location": resolved, "days": days, "error": None}

    except Exception as exc:
        logger.warning("get_forecast failed for '%s': %s", location, exc)
        return {"location": location, "days": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    async def _test() -> None:
        print("=== weather_api smoke-test ===\n")
        for place in ["Pune", "Ludhiana", "ThisPlaceDoesNotExist99"]:
            result = await get_forecast(place)
            print(f"Query : {place}")
            if result["error"]:
                print(f"Error : {result['error']}")
            else:
                print(f"Resolved : {result['location']}")
                for day in result["days"]:
                    print(
                        f"  {day['date']}  max={day['max_temp']}°C  "
                        f"min={day['min_temp']}°C  "
                        f"rain={day['rain_mm']}mm  "
                        f"wind={day['wind_kmh']}km/h"
                    )
            print()

    asyncio.run(_test())