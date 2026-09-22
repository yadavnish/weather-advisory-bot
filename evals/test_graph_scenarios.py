"""End-to-end scenario tests that run the ACTUAL compiled LangGraph, with
weather.fetch_weather / geocode and semantic_matcher.classify monkeypatched
so the suite runs offline and deterministically in CI.

This is the eval suite referenced in the README: each case pins a question +
weather snapshot to an expected primary_sop_id (or expected fallback), so
"add the 11th SOP without changing application logic" is something you can
demonstrate by adding a case here, not just claim in prose.

Run with:  python3 -m pytest evals/test_graph_scenarios.py -v
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import graph as graph_module
from backend.models import SessionState, WeatherSnapshot
from backend.semantic_matcher import SemanticResult
from backend.sop_engine import load_sops

SOPS_PATH = Path(__file__).resolve().parent.parent / "sops.yaml"
SOPS = load_sops(SOPS_PATH)


def snap(**overrides) -> WeatherSnapshot:
    defaults = dict(
        location_name="Bengaluru, Karnataka, India",
        latitude=12.97,
        longitude=77.59,
        fetched_at=datetime.now(timezone.utc),
        temperature_2m=28.0,
        wind_speed_10m=10.0,
        wind_gusts_10m=15.0,
        precipitation=0.0,
        precipitation_probability=10.0,
        precipitation_sum=0.0,
        uv_index=4.0,
        humidity=50.0,
    )
    defaults.update(overrides)
    return WeatherSnapshot(**defaults)


async def run_case(question, snapshot, semantic_result=None, semantic_results_by_sop=None, location_hint="Bengaluru"):
    """semantic_result: applied to EVERY semantic SOP (simple cases).
    semantic_results_by_sop: {sop_id: SemanticResult} for cases where
    multiple semantic SOPs are in play and must respond differently."""
    g = graph_module.build_graph()
    session = SessionState()
    by_sop = semantic_results_by_sop or {}

    async def fake_geocode(hint):
        return snapshot.latitude, snapshot.longitude, snapshot.location_name

    async def fake_fetch_weather(lat, lon, name):
        return snapshot

    async def fake_classify(q, sop, snap_):
        if sop.id in by_sop:
            return by_sop[sop.id]
        if semantic_result is not None:
            return semantic_result
        return SemanticResult(applies=False, confidence=0.0, summary="")

    with patch.object(graph_module, "geocode", fake_geocode), \
         patch.object(graph_module, "fetch_weather", fake_fetch_weather), \
         patch.object(graph_module, "classify", fake_classify):
        result = await g.ainvoke(
            {
                "question": question,
                "location_hint": location_hint,
                "session": session,
                "sops": SOPS,
            }
        )
    return result["response"]


@pytest.mark.asyncio
async def test_high_wind_cycling_blocks():
    resp = await run_case("Is it safe to cycle today?", snap(wind_speed_10m=45.0))
    assert resp.primary_sop_id == "SOP-002"
    assert resp.grounded
    assert "45.0" in resp.answer


@pytest.mark.asyncio
async def test_calm_travel_day_low_severity():
    resp = await run_case(
        "Is it a good day to drive to the next city?",
        snap(precipitation_probability=5.0, wind_speed_10m=8.0, uv_index=3.0),
    )
    assert resp.primary_sop_id == "SOP-011"
    assert resp.grounded


@pytest.mark.asyncio
async def test_extreme_weather_override_supersedes_category():
    resp = await run_case(
        "Should I take my kid to the park?",
        snap(precipitation_sum=90.0, wind_gusts_10m=30.0, temperature_2m=39.0),
    )
    # SOP-010 (severity 5, systemic_override) must outrank SOP-005 (severity 3, temp threshold)
    assert resp.primary_sop_id == "SOP-010"
    assert "SOP-005" in resp.other_matched_sop_ids
    assert resp.grounded


@pytest.mark.asyncio
async def test_semantic_picnic_case_uses_llm_summary_not_freewritten_advice():
    resp = await run_case(
        "Is today good for a picnic in the park?",
        snap(temperature_2m=24.0, wind_speed_10m=6.0, precipitation=0.0, uv_index=3.0),
        semantic_results_by_sop={
            "SOP-009": SemanticResult(
                applies=True,
                confidence=0.9,
                summary="temperature is 24.0C, wind is calm at 6.0 km/h, no rain, UV index 3.0",
            ),
            "SOP-012": SemanticResult(applies=False, confidence=0.8, summary="not a sustained-effort activity"),
        },
    )
    assert resp.primary_sop_id == "SOP-009"
    assert "24.0" in resp.answer  # llm_summary got slotted into the template
    assert resp.grounded


@pytest.mark.asyncio
async def test_no_matching_sop_returns_honest_fallback():
    resp = await run_case(
        "Will there be a solar eclipse today?",
        snap(),
    )
    assert resp.primary_sop_id is None
    assert not resp.grounded
    assert resp.fallback_reason == "no_policy_match"


@pytest.mark.asyncio
async def test_geocode_failure_routes_to_honest_fallback():
    g = graph_module.build_graph()
    session = SessionState()

    async def failing_geocode(hint):
        from backend.weather import WeatherFetchError

        raise WeatherFetchError("no location found for 'Nowhereville'")

    with patch.object(graph_module, "geocode", failing_geocode):
        result = await g.ainvoke(
            {
                "question": "is it safe to cycle?",
                "location_hint": "Nowhereville",
                "session": session,
                "sops": SOPS,
            }
        )
    resp = result["response"]
    assert not resp.grounded
    assert resp.fallback_reason.startswith("location_error")


@pytest.mark.asyncio
async def test_session_reuses_location_on_followup():
    """Second turn with no location_hint should reuse session state instead
    of asking again."""
    g = graph_module.build_graph()
    session = SessionState()
    s = snap(wind_speed_10m=10.0)

    async def fake_geocode(hint):
        return s.latitude, s.longitude, s.location_name

    async def fake_fetch_weather(lat, lon, name):
        return s

    with patch.object(graph_module, "geocode", fake_geocode), \
         patch.object(graph_module, "fetch_weather", fake_fetch_weather):
        await g.ainvoke(
            {"question": "is it safe to cycle?", "location_hint": "Bengaluru", "session": session, "sops": SOPS}
        )
        result2 = await g.ainvoke(
            {"question": "what about walking?", "location_hint": None, "session": session, "sops": SOPS}
        )
    resp2 = result2["response"]
    assert resp2.fallback_reason != "no_location"


if __name__ == "__main__":
    import subprocess

    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
