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

import httpx

from .models import WeatherSnapshot

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherFetchError(Exception):
    pass


async def geocode(place_name: str) -> tuple[float, float, str]:
    normalized = place_name.strip().lower()

    # Open-Meteo can resolve "Bangalore" to Bangalore Town, Pakistan.
    # Use the well-known Bengaluru coordinates when the user explicitly
    # asks for Bangalore/Bengaluru.
    if normalized in {"bangalore", "bengaluru"}:
        return 12.9716, 77.5946, "Bengaluru, Karnataka, India"

    params = {"name": place_name, "count": 10, "language": "en", "format": "json"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(GEOCODE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise WeatherFetchError(f"geocoding request failed: {e}") from e

    results = data.get("results") or []
    if not results:
        raise WeatherFetchError(f"no location found for '{place_name}'")

    top = results[0]
    resolved_name = ", ".join(
        part for part in [top.get("name"), top.get("admin1"), top.get("country")] if part
    )
    return top["latitude"], top["longitude"], resolved_name


async def fetch_weather(latitude: float, longitude: float, location_name: str) -> WeatherSnapshot:
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
        "daily": ",".join(["precipitation_sum", "precipitation_probability_max"]),
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(3):
                resp = await client.get(FORECAST_URL, params=params)

                if resp.status_code == 429:
                    if attempt == 2:
                        resp.raise_for_status()

                    await asyncio.sleep(2 * (attempt + 1))
                    continue

                resp.raise_for_status()
                data = resp.json()
                break

    except (httpx.HTTPError, ValueError) as e:
        raise WeatherFetchError(f"weather request failed: {e}") from e
    
    try:
        current = data["current"]
        daily = data["daily"]
        return WeatherSnapshot(
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
    except (KeyError, IndexError) as e:
        raise WeatherFetchError(f"unexpected Open-Meteo response shape: {e}") from e
