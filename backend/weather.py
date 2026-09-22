"""
Open-Meteo client: geocoding + current/forecast weather.

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

_weather_cache: dict[
    tuple[float, float],
    tuple[float, WeatherSnapshot],
] = {}

CACHE_TTL_SECONDS = 300


class WeatherFetchError(Exception):
    pass


async def geocode(place_name: str) -> tuple[float, float, str]:
    """
    Resolve a city/place name into latitude, longitude, and a readable
    location name using Open-Meteo geocoding.
    """

    normalized = place_name.strip().lower()

    # Open-Meteo can sometimes resolve "Bangalore" to Bangalore Town,
    # Pakistan. Handle Bangalore/Bengaluru explicitly.
    if normalized in {"bangalore", "bengaluru"}:
        return (
            12.9716,
            77.5946,
            "Bengaluru, Karnataka, India",
        )

    params = {
        "name": place_name,
        "count": 10,
        "language": "en",
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                GEOCODE_URL,
                params=params,
            )

            resp.raise_for_status()
            data = resp.json()

    except (httpx.HTTPError, ValueError) as e:
        raise WeatherFetchError(
            f"geocoding request failed: {e}"
        ) from e

    results = data.get("results") or []

    if not results:
        raise WeatherFetchError(
            f"no location found for '{place_name}'"
        )

    top = results[0]

    resolved_name = ", ".join(
        part
        for part in [
            top.get("name"),
            top.get("admin1"),
            top.get("country"),
        ]
        if part
    )

    try:
        return (
            top["latitude"],
            top["longitude"],
            resolved_name,
        )
    except KeyError as e:
        raise WeatherFetchError(
            f"unexpected geocoding response: missing {e}"
        ) from e


async def fetch_weather(
    latitude: float,
    longitude: float,
    location_name: str,
) -> WeatherSnapshot:
    """
    Fetch current and daily forecast weather from Open-Meteo.

    Uses a short in-memory cache to reduce repeated calls and retries
    HTTP 429 responses before failing honestly.
    """

    cache_key = (
        round(latitude, 4),
        round(longitude, 4),
    )

    # ---------------------------------------------------------
    # Check short-lived cache
    # ---------------------------------------------------------

    cached = _weather_cache.get(cache_key)

    if cached:
        cached_at, cached_snapshot = cached

        if time.time() - cached_at < CACHE_TTL_SECONDS:
            return cached_snapshot

    # ---------------------------------------------------------
    # Open-Meteo request parameters
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Fetch weather with retry for HTTP 429
    # ---------------------------------------------------------

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:

            data = None

            for attempt in range(3):

                resp = await client.get(
                    FORECAST_URL,
                    params=params,
                )

                # Open-Meteo rate limit
                if resp.status_code == 429:

                    retry_after = resp.headers.get(
                        "Retry-After"
                    )

                    # Last attempt: let raise_for_status()
                    # generate the final HTTP error.
                    if attempt == 2:
                        resp.raise_for_status()

                    try:
                        delay = (
                            float(retry_after)
                            if retry_after
                            else 5.0
                        )
                    except ValueError:
                        delay = 5.0

                    # Don't wait longer than 15 seconds.
                    await asyncio.sleep(
                        min(delay, 15.0)
                    )

                    continue

                # Any other HTTP error
                resp.raise_for_status()

                data = resp.json()

                break

            if data is None:
                raise WeatherFetchError(
                    "weather request failed after retries"
                )

    except (httpx.HTTPError, ValueError) as e:
        raise WeatherFetchError(
            f"weather request failed: {e}"
        ) from e

    # ---------------------------------------------------------
    # Parse Open-Meteo response
    # ---------------------------------------------------------

    try:
        current = data["current"]
        daily = data["daily"]

        snapshot = WeatherSnapshot(
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            fetched_at=datetime.now(timezone.utc),

            temperature_2m=current[
                "temperature_2m"
            ],

            wind_speed_10m=current[
                "wind_speed_10m"
            ],

            wind_gusts_10m=current[
                "wind_gusts_10m"
            ],

            precipitation=current[
                "precipitation"
            ],

            precipitation_probability=daily[
                "precipitation_probability_max"
            ][0],

            precipitation_sum=daily[
                "precipitation_sum"
            ][0],

            uv_index=current.get(
                "uv_index",
                0.0,
            ),

            humidity=current[
                "relative_humidity_2m"
            ],
        )

        # Save successful result in cache.
        _weather_cache[cache_key] = (
            time.time(),
            snapshot,
        )

        return snapshot

    except (KeyError, IndexError, TypeError) as e:
        raise WeatherFetchError(
            f"unexpected Open-Meteo response shape: {e}"
        ) from e