"""Fully offline tests for the deterministic parts of the engine.

Run with:  python3 -m pytest evals/test_sop_engine.py -v
No network, no LLM, no API key needed -- this is exactly the "zero
hallucination risk" claim made about threshold SOPs, backed by a real test.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import sop_engine
from backend.models import WeatherSnapshot

SOPS_PATH = Path(__file__).resolve().parent.parent / "sops.yaml"


def make_snapshot(**overrides) -> WeatherSnapshot:
    defaults = dict(
        location_name="Test City",
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


def test_load_sops():
    sops = sop_engine.load_sops(SOPS_PATH)
    assert len(sops) == 13
    ids = {s.id for s in sops}
    assert "SOP-010" in ids
    assert all(s.advice_template for s in sops)


def test_high_wind_blocks_cycling():
    sops = sop_engine.load_sops(SOPS_PATH)
    sop002 = next(s for s in sops if s.id == "SOP-002")
    snap = make_snapshot(wind_speed_10m=45.0)
    m = sop_engine.match_threshold_sop(sop002, "is it safe to cycle today?", snap)
    assert m.matched
    text = sop_engine.render_template(sop002, snap)
    assert "45.0" in text


def test_low_wind_does_not_block_cycling():
    sops = sop_engine.load_sops(SOPS_PATH)
    sop002 = next(s for s in sops if s.id == "SOP-002")
    snap = make_snapshot(wind_speed_10m=12.0)
    m = sop_engine.match_threshold_sop(sop002, "is it safe to cycle today?", snap)
    assert not m.matched


def test_keyword_gate_prevents_false_positive():
    """A travel SOP should not fire on a cycling question even if the
    numeric condition happens to be true -- keyword gate must hold."""
    sops = sop_engine.load_sops(SOPS_PATH)
    sop003 = next(s for s in sops if s.id == "SOP-003")  # travel, precip%>=70
    snap = make_snapshot(precipitation_probability=90.0)
    m = sop_engine.match_threshold_sop(sop003, "should I take my kid to the park?", snap)
    assert not m.matched, "keyword gate should block unrelated question"


def test_systemic_override_fires_on_extreme_rain():
    sops = sop_engine.load_sops(SOPS_PATH)
    sop010 = next(s for s in sops if s.id == "SOP-010")
    snap = make_snapshot(precipitation_sum=80.0, wind_gusts_10m=20.0)
    m = sop_engine.match_systemic_override(sop010, snap)
    assert m.matched


def test_systemic_override_or_condition_on_wind_alone():
    sops = sop_engine.load_sops(SOPS_PATH)
    sop010 = next(s for s in sops if s.id == "SOP-010")
    snap = make_snapshot(precipitation_sum=0.0, wind_gusts_10m=70.0)
    m = sop_engine.match_systemic_override(sop010, snap)
    assert m.matched, "OR clause: extreme wind alone should also trigger the override"


def test_rank_and_select_prefers_higher_severity():
    sops = sop_engine.load_sops(SOPS_PATH)
    sop006 = next(s for s in sops if s.id == "SOP-006")  # severity 2
    sop005 = next(s for s in sops if s.id == "SOP-005")  # severity 3
    from backend.models import SOPMatch

    matches = [
        SOPMatch(sop=sop006, matched=True, rationale="x"),
        SOPMatch(sop=sop005, matched=True, rationale="y"),
    ]
    primary, others = sop_engine.rank_and_select(matches)
    assert primary.sop.id == "SOP-005"
    assert others[0].sop.id == "SOP-006"


def test_rank_and_select_tie_prefers_override_type():
    from backend.models import SOPMatch

    sops = sop_engine.load_sops(SOPS_PATH)
    sop010 = next(s for s in sops if s.id == "SOP-010")  # systemic_override, sev 5
    fake_threshold = sop_engine.SOP(
        id="FAKE-SEV5", category="x", match_type="threshold", severity=5, advice_template="x"
    )
    matches = [
        SOPMatch(sop=fake_threshold, matched=True, rationale="tie test"),
        SOPMatch(sop=sop010, matched=True, rationale="tie test"),
    ]
    primary, _ = sop_engine.rank_and_select(matches)
    assert primary.sop.id == "SOP-010", "systemic_override must win severity ties"


def test_no_match_returns_none():
    primary, others = sop_engine.rank_and_select([])
    assert primary is None
    assert others == []


def test_numeric_tokens_grounding_set():
    snap = make_snapshot(wind_speed_10m=45.0)
    tokens = snap.numeric_tokens()
    assert "45.0" in tokens
    assert "45" in tokens
    assert "9999" not in tokens


if __name__ == "__main__":
    import subprocess

    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
