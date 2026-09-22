"""Core data models shared across the SOP engine and graph.

Kept deliberately dependency-free (dataclasses only) so the matching logic
in sop_engine.py can be unit tested with zero network/LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class WeatherSnapshot:
    """A single point-in-time weather reading pulled from Open-Meteo.

    Every numeric field that can appear in an advice_template MUST live here.
    The grounding validator (validate_grounding in graph.py) checks the final
    response text against exactly these values -- nothing else counts as
    "traceable".
    """

    location_name: str
    latitude: float
    longitude: float
    fetched_at: datetime
    temperature_2m: float
    wind_speed_10m: float
    wind_gusts_10m: float
    precipitation: float
    precipitation_probability: float
    precipitation_sum: float  # daily total, used by the systemic override
    uv_index: float
    humidity: float

    def as_dict(self) -> dict[str, Any]:
        d = {
            "temperature_2m": self.temperature_2m,
            "wind_speed_10m": self.wind_speed_10m,
            "wind_gusts_10m": self.wind_gusts_10m,
            "precipitation": self.precipitation,
            "precipitation_probability": self.precipitation_probability,
            "precipitation_sum": self.precipitation_sum,
            "uv_index": self.uv_index,
            "humidity": self.humidity,
        }
        return d

    def numeric_tokens(self) -> set[str]:
        """String forms of every number that is legitimately groundable.

        Includes a couple of rounding variants (int vs 1-decimal) since the
        template renderer may format either way -- the validator should not
        false-positive on formatting, only on genuinely invented numbers.
        """
        tokens: set[str] = set()
        for v in self.as_dict().values():
            tokens.add(str(v))
            tokens.add(str(round(v)))
            tokens.add(f"{v:.1f}")
        return tokens


@dataclass
class SOP:
    id: str
    category: str
    match_type: str  # "threshold" | "semantic" | "systemic_override"
    severity: int
    advice_template: str
    conditions: list[dict] = field(default_factory=list)
    keywords_any: list[str] = field(default_factory=list)
    description: Optional[str] = None
    id_note: Optional[str] = None


@dataclass
class SOPMatch:
    sop: SOP
    matched: bool
    rationale: str
    llm_summary: Optional[str] = None  # only set for semantic matches
    confidence: Optional[float] = None


@dataclass
class SessionState:
    """Minimal structured memory carried across turns of one conversation.

    Deliberately NOT raw chat-history replay -- only the fields that a
    follow-up question actually needs to reuse.
    """

    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    last_snapshot: Optional[WeatherSnapshot] = None
    last_sop_ids: list[str] = field(default_factory=list)


@dataclass
class BotResponse:
    answer: str
    primary_sop_id: Optional[str]
    other_matched_sop_ids: list[str]
    weather_snapshot: Optional[dict]
    grounded: bool
    fallback_reason: Optional[str] = None
