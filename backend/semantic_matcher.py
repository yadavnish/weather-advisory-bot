"""Semantic SOP classifier using Google Gemini.

The LLM only decides whether a semantic SOP applies and provides a
short factual weather summary. It does not write the final advice.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from google import genai

from .models import SOP, WeatherSnapshot


_CLASSIFIER_SYSTEM = """You are a semantic policy applicability classifier
for a weather-advisory policy engine.

You will be given:
1. A user's question.
2. A candidate policy's description.
3. The actual weather readings.

Your job is ONLY to decide whether the candidate policy applies to the
user's intent.

Rules:
- Match the meaning of the user's request, not just exact keywords.
- Treat paraphrases as equivalent.
- If the policy describes the type of activity the user is asking about,
  it can apply even without a numeric threshold.
- Use weather readings only to describe current conditions.
- Do NOT invent thresholds.
- Do NOT create new weather rules.
- Do NOT write the final recommendation.
- Do NOT give safety advice.

Examples:
- "Is today good for a picnic?" matches a picnic/outdoor-gathering policy.
- "Is today suitable for a picnic in Bangalore?" matches a picnic/outdoor-gathering policy.
- "Would today be suitable for spending the day outside?" can match the
  same policy.
- "Should I have a picnic today?" matches a picnic/outdoor-gathering policy.
- "Is this a good day for an outdoor gathering?" matches a
  picnic/outdoor-gathering policy.
- "Can I go cycling?" should match a cycling policy, not a picnic policy.

Return ONLY valid JSON:

{
  "applies": true,
  "confidence": 0.0,
  "summary": "one factual sentence describing the relevant weather conditions"
}

The confidence must be between 0.0 and 1.0.
"""


@dataclass
class SemanticResult:
    applies: bool
    confidence: float
    summary: str
    raw_error: str | None = None


def _deterministic_picnic_fallback(
    question: str,
    sop: SOP,
    snapshot: WeatherSnapshot,
) -> SemanticResult | None:
    if sop.id != "SOP-009":
        return None

    picnic_terms = [
        "picnic",
        "outdoor gathering",
        "spend the day outside",
        "spending the day outside",
        "outdoors today",
        "outside today",
    ]
    if not any(term in question.lower() for term in picnic_terms):
        return None

    precip_prob = snapshot.precipitation_probability
    precip_sum = snapshot.precipitation_sum
    wind_gusts = snapshot.wind_gusts_10m
    humidity = snapshot.humidity
    uv = snapshot.uv_index

    concerns = []
    verdict = "look good"

    if precip_prob is not None and precip_prob >= 60:
        verdict = "risky"
        concerns.append(f"a {precip_prob}% chance of rain")
    if precip_sum is not None and precip_sum > 0.5:
        verdict = "risky"
        concerns.append(f"expected rainfall of {precip_sum}mm")
    if wind_gusts is not None and wind_gusts > 30:
        if verdict == "look good":
            verdict = "a bit uncertain"
        concerns.append(f"gusty wind up to {wind_gusts} km/h")
    if humidity is not None and humidity > 85:
        if verdict == "look good":
            verdict = "a bit uncertain"
        concerns.append(f"high humidity at {humidity}%")
    if uv is not None and uv > 8:
        if verdict == "look good":
            verdict = "a bit uncertain"
        concerns.append(f"a high UV index of {uv}")

    if concerns:
        summary = (
            f"Conditions {verdict} for a picnic today, mainly due to "
            + ", ".join(concerns)
            + " (deterministic fallback estimate, semantic model unavailable)."
        )
    else:
        summary = (
            "Conditions look favorable for a picnic today - comfortable "
            "temperature, low rain risk, and manageable wind and humidity "
            "(deterministic fallback estimate, semantic model unavailable)."
        )

    return SemanticResult(
        applies=True,
        confidence=0.6,
        summary=summary,
        raw_error=None,
    )

async def classify(
    question: str,
    sop: SOP,
    snapshot: WeatherSnapshot,
) -> SemanticResult:
    """Classify whether a semantic SOP applies."""

    assert sop.match_type == "semantic"

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return SemanticResult(
            applies=False,
            confidence=0.0,
            summary="",
            raw_error="GEMINI_API_KEY is not set",
        )

    client = genai.Client(api_key=api_key)

    user_content = json.dumps(
        {
            "question": question,
            "policy_id": sop.id,
            "policy_description": sop.description,
            "weather_readings": snapshot.as_dict(),
        },
        indent=2,
    )

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                _CLASSIFIER_SYSTEM
                                + "\n\nINPUT:\n"
                                + user_content
                            )
                        }
                    ],
                }
            ],
        )

        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "", 1)
            text = text.replace("```", "")
            text = text.strip()

        parsed = json.loads(text)

        return SemanticResult(
            applies=bool(parsed["applies"]),
            confidence=float(parsed.get("confidence", 0.5)),
            summary=str(parsed.get("summary", "")),
        )

    except Exception as e:
        print(f"SEMANTIC CLASSIFIER ERROR: {e}")
        fallback = _deterministic_picnic_fallback(question, sop, snapshot)
        if fallback is not None:
            return fallback
        return SemanticResult(
            applies=False,
            confidence=0.0,
            summary="",
            raw_error="gemini_unavailable",
        )
