"""Open-Meteo client: geocoding + current/forecast weather.

Open-Meteo needs no API key. Two calls per turn:
  1. geocode(place_name)   -> lat/lon
  2. fetch_weather(lat,lon) -> WeatherSnapshot

Both raise WeatherFetchError on failure so the graph can route to the
honest_fallback node instead of silently returning stale/fake data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import time

import httpx

from .models import WeatherSnapshot

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_weather_cache: dict[tuple[float, float], tuple[float, WeatherSnapshot]] = {}
CACHE_TTL_SECONDS = 300


class WeatherFetchError(Exception):
    pass


async def fetch_weather(
    latitude: float,
    longitude: float,
    location_name: str,
) -> WeatherSnapshot:

    cache_key = (round(latitude, 4), round(longitude, 4))
    cached = _weather_cache.get(cache_key)

    if cached:
        cached_at, cached_snapshot = cached
        if time.time() - cached_at < CACHE_TTL_SECONDS:
            return cached_snapshot

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(
            [
                "temperature_2m",
                "wind_speed_10m",
                "wind_gusts_10m",
                "precipitation",
                "relative_humidity_2m",
                "uv_index",
            ]
        ),
        "daily": ",".join(
            [
                "precipitation_sum",
                "precipitation_probability_max",
            ]
        ),
        "timezone": "auto",
        "forecast_days": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(3):
                resp = await client.get(FORECAST_URL, params=params)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")

                    if attempt == 2:
                        resp.raise_for_status()

                    try:
                        delay = float(retry_after) if retry_after else 5.0
                    except ValueError:
                        delay = 5.0

                    await asyncio.sleep(min(delay, 15.0))
                    continue

                resp.raise_for_status()
                data = resp.json()
                break

    except (httpx.HTTPError, ValueError) as e:
        raise WeatherFetchError(f"weather request failed: {e}") from e

    try:
        current = data["current"]
        daily = data["daily"]

        snapshot = WeatherSnapshot(
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            fetched_at=datetime.now(timezone.utc),
            temperature_2m=current["temperature_2m"],
            wind_speed_10m=current["wind_speed_10m"],
            wind_gusts_10m=current["wind_gusts_10m"],
            precipitation=current["precipitation"],
            precipitation_probability=daily["precipitation_probability_max"][0],
            precipitation_sum=daily["precipitation_sum"][0],
            uv_index=current.get("uv_index", 0.0),
            humidity=current["relative_humidity_2m"],
        )

        _weather_cache[cache_key] = (time.time(), snapshot)

        return snapshot

    except (KeyError, IndexError) as e:
        raise WeatherFetchError(
            f"unexpected Open-Meteo response shape: {e}"
        ) from e