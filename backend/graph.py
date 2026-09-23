"""LangGraph wiring for the weather-advisory bot.

    START
      -> resolve_location   (geocode; failure -> honest_fallback)
      -> fetch_weather_node (Open-Meteo; failure -> honest_fallback)
      -> check_systemic_override
      -> match_threshold_sops
      -> match_semantic_sops
      -> rank_and_select_node
           -> no SOP matched -> no_policy_fallback
           -> else           -> compose_answer   (template render, not free-gen)
      -> validate_grounding  (reject/repair if a number can't be traced)
      -> END

honest_fallback / no_policy_fallback are separate terminal nodes so the
failure path is real branching, not a swallowed exception inside one node.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import sop_engine
from .models import SOP, BotResponse, SessionState, SOPMatch, WeatherSnapshot
from .semantic_matcher import classify
from .weather import WeatherFetchError, fetch_weather, geocode


class GraphState(TypedDict, total=False):
    question: str
    location_hint: Optional[str]
    session: SessionState
    sops: list[SOP]

    latitude: float
    longitude: float
    location_name: str
    snapshot: WeatherSnapshot

    override_match: Optional[SOPMatch]
    threshold_matches: list[SOPMatch]
    semantic_matches: list[SOPMatch]
    all_matches: list[SOPMatch]

    primary: Optional[SOPMatch]
    others: list[SOPMatch]
    answer_text: str

    response: BotResponse
    error: Optional[str]


async def resolve_location(state: GraphState) -> GraphState:
    session = state["session"]
    hint = state.get("location_hint")

    # Reuse session location if the user didn't give a new one this turn.
    if not hint and session.latitude is not None:
        return {
            **state,
            "latitude": session.latitude,
            "longitude": session.longitude,
            "location_name": session.location_name,
        }

    if not hint:
        return {**state, "error": "no_location", "location_name": ""}

    try:
        lat, lon, resolved_name = await geocode(hint)
        return {
            **state,
            "latitude": lat,
            "longitude": lon,
            "location_name": resolved_name,
        }
    except WeatherFetchError as e:
        return {**state, "error": f"location_error: {e}"}


async def fetch_weather_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    try:
        snapshot = await fetch_weather(
            state["latitude"],
            state["longitude"],
            state["location_name"],
        )

        # Persist the resolved location as soon as it's confirmed good, even
        # if no SOP ends up matching this turn's question -- a follow-up
        # question shouldn't have to re-supply the location.
        session = state["session"]
        session.location_name = state["location_name"]
        session.latitude = state["latitude"]
        session.longitude = state["longitude"]

        return {**state, "snapshot": snapshot}

    except WeatherFetchError as e:
        return {**state, "error": f"weather_error: {e}"}


async def check_systemic_override(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    override_sops = [
        s for s in state["sops"]
        if s.match_type == "systemic_override"
    ]

    match = None

    for sop in override_sops:
        m = sop_engine.match_systemic_override(
            sop,
            state["snapshot"],
        )

        if m.matched:
            match = m
            break

    return {**state, "override_match": match}


async def match_threshold_sops(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    now_hour = datetime.now().hour

    threshold_sops = [
        s for s in state["sops"]
        if s.match_type == "threshold"
    ]

    matches = [
        sop_engine.match_threshold_sop(
            sop,
            state["question"],
            state["snapshot"],
            now_hour,
        )
        for sop in threshold_sops
    ]

    return {**state, "threshold_matches": matches}


async def match_semantic_sops(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    semantic_sops = [
        s for s in state["sops"]
        if s.match_type == "semantic"
    ]

    matches: list[SOPMatch] = []
    errors: list[str] = []

    for sop in semantic_sops:
        result = await classify(
            state["question"],
            sop,
            state["snapshot"],
        )

        # IMPORTANT:
        # A classifier/API failure is NOT the same thing as
        # "no semantic SOP matched".
        #
        # For example, Gemini quota exhaustion should not cause
        # the system to incorrectly report "no policy match".
        #
        # We no longer bail out on the first failure: one semantic SOP
        # may have a deterministic fallback (e.g. picnic) that succeeds
        # even while another semantic SOP has no fallback and fails. We
        # only report "unavailable" for the whole batch if nothing
        # could be resolved at all.
        if result.raw_error:
            errors.append(f"{sop.id}: {result.raw_error}")
            continue

        matches.append(
            SOPMatch(
                sop=sop,
                matched=(
                    result.applies
                    and result.confidence >= 0.5
                ),
                rationale=(
                    f"llm classifier confidence="
                    f"{result.confidence}"
                ),
                llm_summary=result.summary or None,
                confidence=result.confidence,
            )
        )

     # A semantic-classification failure should only escalate to a
    # visible error if there is truly nothing else to answer with.
    # If a threshold SOP or the systemic override already matched,
    # that real, grounded answer must not be discarded just because
    # an unrelated semantic SOP couldn't be checked.
    has_other_signal = bool(state.get("threshold_matches")) or bool(
        state.get("override_match")
    )

    if errors and not matches and not has_other_signal:
        return {
            **state,
            "error": (
                "semantic_model_unavailable: "
                + "; ".join(errors)
            ),
        }

    return {
        **state,
        "semantic_matches": matches,
    }


async def rank_and_select_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    all_matches = (
        list(state.get("threshold_matches", []))
        + list(state.get("semantic_matches", []))
    )

    override = state.get("override_match")

    if override is not None:
        all_matches.append(override)

    primary, others = sop_engine.rank_and_select(
        all_matches
    )

    return {
        **state,
        "all_matches": all_matches,
        "primary": primary,
        "others": others,
    }


async def compose_answer(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    primary = state.get("primary")
    override = state.get("override_match")
    snapshot = state["snapshot"]

    if primary is None:
        return {
            **state,
            "error": "no_policy_match",
        }

    primary_text = sop_engine.render_template(
        primary.sop,
        snapshot,
        primary.llm_summary,
    )

    # Systemic override gets prepended even when it wasn't picked as primary
    # by rank_and_select (e.g. exact severity tie already resolved it, but
    # this guards the case where a non-override SOP of equal severity sorted
    # first due to future rule changes).
    if (
        override is not None
        and override.matched
        and primary.sop.id != override.sop.id
    ):
        override_text = sop_engine.render_template(
            override.sop,
            snapshot,
        )

        primary_text = (
            f"{override_text}\n\n{primary_text}"
        )

    others = [
        m
        for m in state.get("others", [])
        if m.sop.id != primary.sop.id
    ]

    if others:
        disclosures = "; ".join(
            f"{m.sop.id} (severity {m.sop.severity})"
            for m in others
        )

        primary_text += (
            "\n\n"
            f"[Also applicable, not primary: {disclosures}]"
        )

    return {
        **state,
        "answer_text": primary_text,
    }


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


async def validate_grounding(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    answer = state["answer_text"]
    snapshot = state["snapshot"]

    allowed = snapshot.numeric_tokens()

    # Policy-defined constants (e.g. "40 km/h threshold") are legitimate --
    # they're audited data from sops.yaml, not model output. Union in the
    # constants from every SOP that actually matched this turn.
    matched_sops = [state["primary"].sop]

    if state.get("override_match"):
        matched_sops.append(
            state["override_match"].sop
        )

    matched_sops.extend(
        m.sop
        for m in state.get("others", [])
    )

    for sop in matched_sops:
        allowed |= (
            sop_engine.collect_condition_value_tokens(sop)
        )

    # Severity numbers and SOP-id numbers are meta, not weather facts -- skip.
    skip_context = re.compile(
        r"(severity|SOP-)\s*\d*"
    )

    scrubbed = skip_context.sub(
        "",
        answer,
    )

    found_numbers = set(
        _NUMBER_RE.findall(scrubbed)
    )

    ungrounded = {
        n for n in found_numbers
        if n not in allowed
    }

    if ungrounded:
        return {
            **state,
            "error": (
                "grounding_failed: "
                f"numbers {ungrounded} not in weather snapshot"
            ),
        }

    return state


def _terminal_response(state: GraphState) -> GraphState:
    error = state.get("error")
    session: SessionState = state["session"]

    if error is None:
        primary = state["primary"]
        others = state.get("others", [])

        session.location_name = state["location_name"]
        session.latitude = state["latitude"]
        session.longitude = state["longitude"]
        session.last_snapshot = state["snapshot"]

        session.last_sop_ids = (
            [primary.sop.id]
            + [m.sop.id for m in others]
        )

        response = BotResponse(
            answer=state["answer_text"],
            primary_sop_id=primary.sop.id,
            other_matched_sop_ids=[
                m.sop.id for m in others
            ],
            weather_snapshot=state["snapshot"].as_dict(),
            grounded=True,
        )

        return {
            **state,
            "response": response,
        }

    if error == "no_location":
        msg = (
            "I need a location to check conditions - "
            "which city or place?"
        )

    elif error.startswith("location_error"):
        msg = (
            "I couldn't resolve that location, so I can't "
            "give you a verified answer. "
            f"({error})"
        )

    elif error.startswith("weather_error"):
        msg = (
            "I couldn't fetch live weather data, so I can't "
            "give you a verified answer. "
            f"({error})"
        )

    elif error.startswith("semantic_model_unavailable"):
        msg = (
            "The semantic policy classifier is temporarily "
            "unavailable, so I can't safely determine which "
            "weather policy applies. I won't guess."
        )

    elif error == "no_policy_match":
        msg = (
            "We don't have specific guidance covering that "
            "scenario yet, so I won't guess."
        )

    elif error.startswith("grounding_failed"):
        msg = (
            "I couldn't produce a fully verified answer for "
            "that (a value didn't trace back to the weather "
            "data), so I'm holding back rather than risk a "
            "wrong number."
        )

    else:
        msg = (
            "Something went wrong and I can't give you a "
            "verified answer right now."
        )

    response = BotResponse(
        answer=msg,
        primary_sop_id=None,
        other_matched_sop_ids=[],
        weather_snapshot=(
            state.get("snapshot").as_dict()
            if state.get("snapshot")
            else None
        ),
        grounded=False,
        fallback_reason=error,
    )

    return {
        **state,
        "response": response,
    }


def _route_after_location(state: GraphState) -> str:
    return (
        "honest_fallback"
        if state.get("error")
        else "fetch_weather"
    )


def _route_after_weather(state: GraphState) -> str:
    return (
        "honest_fallback"
        if state.get("error")
        else "check_systemic_override"
    )


def _route_after_rank(state: GraphState) -> str:
    if state.get("error"):
        return "no_policy_fallback"

    return "compose_answer"


def _route_after_validate(state: GraphState) -> str:
    return (
        "honest_fallback"
        if state.get("error")
        else "finalize"
    )


def build_graph():
    g = StateGraph(GraphState)

    g.add_node(
        "resolve_location",
        resolve_location,
    )

    g.add_node(
        "fetch_weather",
        fetch_weather_node,
    )

    g.add_node(
        "check_systemic_override",
        check_systemic_override,
    )

    g.add_node(
        "match_threshold_sops",
        match_threshold_sops,
    )

    g.add_node(
        "match_semantic_sops",
        match_semantic_sops,
    )

    g.add_node(
        "rank_and_select",
        rank_and_select_node,
    )

    g.add_node(
        "compose_answer",
        compose_answer,
    )

    g.add_node(
        "validate_grounding",
        validate_grounding,
    )

    g.add_node(
        "honest_fallback",
        _terminal_response,
    )

    g.add_node(
        "no_policy_fallback",
        _terminal_response,
    )

    g.add_node(
        "finalize",
        _terminal_response,
    )

    g.set_entry_point(
        "resolve_location"
    )

    g.add_conditional_edges(
        "resolve_location",
        _route_after_location,
        {
            "honest_fallback": "honest_fallback",
            "fetch_weather": "fetch_weather",
        },
    )

    g.add_conditional_edges(
        "fetch_weather",
        _route_after_weather,
        {
            "honest_fallback": "honest_fallback",
            "check_systemic_override": "check_systemic_override",
        },
    )

    g.add_edge(
        "check_systemic_override",
        "match_threshold_sops",
    )

    g.add_edge(
        "match_threshold_sops",
        "match_semantic_sops",
    )

    g.add_edge(
        "match_semantic_sops",
        "rank_and_select",
    )

    g.add_conditional_edges(
        "rank_and_select",
        _route_after_rank,
        {
            "no_policy_fallback": "no_policy_fallback",
            "compose_answer": "compose_answer",
        },
    )

    g.add_edge(
        "compose_answer",
        "validate_grounding",
    )

    g.add_conditional_edges(
        "validate_grounding",
        _route_after_validate,
        {
            "honest_fallback": "honest_fallback",
            "finalize": "finalize",
        },
    )

    g.add_edge(
        "honest_fallback",
        END,
    )

    g.add_edge(
        "no_policy_fallback",
        END,
    )

    g.add_edge(
        "finalize",
        END,
    )

    return g.compile()
