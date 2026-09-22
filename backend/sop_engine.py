"""Everything about SOPs that does NOT require an LLM call.

Design intent (see README for the full argument):
  - threshold SOPs are matched by pure code -> zero hallucination risk,
    testable with plain asserts, no network/LLM dependency at all.
  - semantic SOPs are matched elsewhere (semantic_matcher.py) but rendered
    through the exact same template mechanism as threshold SOPs, so the
    model NEVER free-writes the safety judgment, only a short factual
    summary that gets slotted into a template this module owns.
  - conflict resolution is one named function (rank_and_select), not an
    LLM "pick the best one" call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .models import SOP, SOPMatch, WeatherSnapshot

_OP_FUNCS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def load_sops(path: str | Path) -> list[SOP]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    sops = []
    for entry in raw:
        sops.append(
            SOP(
                id=entry["id"],
                category=entry["category"],
                match_type=entry["match_type"],
                severity=entry["severity"],
                advice_template=entry["advice_template"].strip(),
                conditions=entry.get("conditions", []),
                keywords_any=entry.get("keywords_any", []),
                description=entry.get("description"),
                id_note=entry.get("id_note"),
            )
        )
    return sops


def _keyword_hit(question: str, keywords: list[str]) -> bool:
    if not keywords:
        return True  # no keyword gate -> weather conditions alone decide
    q = question.lower()
    return any(kw.lower() in q for kw in keywords)


def _numeric_conditions_met(conditions: list[dict], snapshot: WeatherSnapshot) -> tuple[bool, list[str]]:
    """Evaluates a list of {field, op, value} conditions as an AND chain,
    with one special case: a literal {"op": "OR"} marker splits the list
    into two OR'd groups (used by SOP-010's rain-OR-wind override).
    Returns (met, [human-readable clause strings]) for logging/rationale.
    """
    if any(c.get("op") == "OR" for c in conditions):
        idx = next(i for i, c in enumerate(conditions) if c.get("op") == "OR")
        left, right = conditions[:idx], conditions[idx + 1 :]
        left_met, left_clauses = _numeric_conditions_met(left, snapshot)
        right_met, right_clauses = _numeric_conditions_met(right, snapshot)
        return (left_met or right_met), left_clauses + ["OR"] + right_clauses

    clauses = []
    for cond in conditions:
        if "hour_between" in cond:
            # Time-of-day gates are evaluated by the caller (needs "now"),
            # so here we just note it was assumed satisfied for offline
            # unit tests; graph.py passes the real check separately.
            clauses.append(f"hour_between({cond['hour_between']})")
            continue
        field_name = cond["field"]
        op = cond["op"]
        threshold = cond["value"]
        actual = getattr(snapshot, field_name)
        ok = _OP_FUNCS[op](actual, threshold)
        clauses.append(f"{field_name}={actual} {op} {threshold} -> {ok}")
        if not ok:
            return False, clauses
    return True, clauses


def match_threshold_sop(sop: SOP, question: str, snapshot: WeatherSnapshot, now_hour: Optional[int] = None) -> SOPMatch:
    assert sop.match_type == "threshold"

    if not _keyword_hit(question, sop.keywords_any):
        return SOPMatch(sop=sop, matched=False, rationale="keyword gate not hit")

    # Handle hour_between explicitly against the real clock if provided.
    for cond in sop.conditions:
        if "hour_between" in cond and now_hour is not None:
            start_h = int(cond["hour_between"][0].split(":")[0])
            end_h = int(cond["hour_between"][1].split(":")[0])
            if not (start_h <= now_hour < end_h):
                return SOPMatch(sop=sop, matched=False, rationale=f"outside hour window {cond['hour_between']}")

    met, clauses = _numeric_conditions_met(sop.conditions, snapshot)
    return SOPMatch(sop=sop, matched=met, rationale="; ".join(clauses))


def match_systemic_override(sop: SOP, snapshot: WeatherSnapshot) -> SOPMatch:
    assert sop.match_type == "systemic_override"
    met, clauses = _numeric_conditions_met(sop.conditions, snapshot)
    return SOPMatch(sop=sop, matched=met, rationale="; ".join(clauses))


def collect_condition_value_tokens(sop: SOP) -> set[str]:
    """Numeric constants that are part of the SOP's own policy definition
    (thresholds like "40 km/h") rather than live weather readings. These are
    legitimate to appear in the rendered text -- they come from sops.yaml,
    an auditable data file, not from the model. The grounding validator
    treats them as pre-approved alongside the live snapshot values.
    """
    tokens: set[str] = set()
    for cond in sop.conditions:
        value = cond.get("value")
        if isinstance(value, (int, float)):
            tokens.add(str(value))
            tokens.add(str(round(value)))
            tokens.add(f"{float(value):.1f}")
    return tokens


_PRIORITY = {"systemic_override": 0, "semantic": 1, "threshold": 2}


def rank_and_select(matches: list[SOPMatch]) -> tuple[Optional[SOPMatch], list[SOPMatch]]:
    """Named, deterministic conflict-resolution algorithm.

    Rank by severity descending; ties broken by match_type priority
    (systemic_override > semantic > threshold). Returns (primary, others)
    where `others` are matched-but-not-primary, kept for disclosure in the
    response ("SOP-004 also applies at severity 2 ...").
    """
    matched = [m for m in matches if m.matched]
    if not matched:
        return None, []
    matched.sort(key=lambda m: (-m.sop.severity, _PRIORITY[m.sop.match_type]))
    primary, *rest = matched
    return primary, rest


def render_template(sop: SOP, snapshot: WeatherSnapshot, llm_summary: Optional[str] = None) -> str:
    values = snapshot.as_dict()
    if llm_summary is not None:
        values["llm_summary"] = llm_summary
        values["rain_wind_summary"] = llm_summary
    elif "{rain_wind_summary}" in sop.advice_template:
        values["rain_wind_summary"] = (
            f"{snapshot.precipitation_sum}mm rainfall today, "
            f"{snapshot.wind_gusts_10m}km/h gusts"
        )
    return sop.advice_template.format(**values)
