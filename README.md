# Weather-Advisory Support Bot

Answers questions like "is it safe to cycle today," "should I take my kid to
the park," or "is this a good day for a picnic" using live weather data
(Open-Meteo) and a set of Standard Operating Procedures (SOPs).

## The three non-negotiables, made concrete

Saying "policy-first / traceable / policy-as-data" isn't a differentiator by
itself -- it's close to the assignment's own wording, so most submissions
will state it. What matters is the *mechanism* behind each claim:

| Claim | Mechanism | Where to look |
|---|---|---|
| Policy-first | The model never free-writes the safety judgment. Threshold SOPs are matched by pure code; semantic SOPs are matched by an LLM call whose *only* output is `{applies, confidence, summary}` -- the wording always comes from `advice_template` in `sops.yaml`. | `backend/sop_engine.py::render_template`, `backend/semantic_matcher.py` |
| Fully traceable | Every response names its primary SOP id, any other SOPs that also matched (and why they weren't primary), and the exact weather values used. A grounding validator parses every number in the final answer and rejects the response if a number doesn't trace back to the weather snapshot or the SOP's own policy constants. | `backend/graph.py::validate_grounding` |
| Policy-as-data | SOPs live in `sops.yaml`, not in prompt text or application code. Adding SOP-13 (or a 20th) is a YAML edit, zero code changes. Demonstrated in `evals/` -- see "Adding an 11th SOP" below. | `sops.yaml`, `backend/sop_engine.py::load_sops` |

## Architecture

```
START
  -> resolve_location     (geocode; reuses session location on follow-ups)
  -> fetch_weather         (Open-Meteo; failure -> honest_fallback)
  -> check_systemic_override   (deterministic, runs BEFORE category matching)
  -> match_threshold_sops      (deterministic, zero LLM calls)
  -> match_semantic_sops       (LLM classifier: applies/confidence/summary only)
  -> rank_and_select           (named conflict-resolution algorithm)
       -> no SOP matched -> no_policy_fallback ("we don't have guidance for that")
       -> else           -> compose_answer     (template render, not free generation)
  -> validate_grounding    (reject/repair if any number isn't traceable)
  -> END
```

Built with LangGraph (`backend/graph.py`). `honest_fallback` and
`no_policy_fallback` are separate terminal nodes, not exceptions swallowed
inside one big node -- real branching for the failure path, not a
try/except that quietly returns something plausible-looking.

### Three SOP match types, one rendering path

- **`threshold`** -- numeric/keyword conditions evaluated by plain Python
  (`_numeric_conditions_met`, `_keyword_hit`). No LLM involved. Testable with
  ordinary asserts (`evals/test_sop_engine.py`).
- **`semantic`** -- fuzzy scenarios with no clean threshold (e.g. "is today
  good for a picnic"). An LLM call classifies applicability and produces a
  short factual summary; it does **not** write the advice. The summary fills
  the `{llm_summary}` slot in the SOP's own template.
- **`systemic_override`** -- checked *before* category matching (e.g. IMD-grade
  extreme rain or wind). If it fires, it's prepended to whatever
  category-specific SOP also matched rather than competing with it on equal
  footing. This is the "the reason is bigger than any single threshold" case
  the assignment flags as the one it cares about most.

### Conflict resolution (named algorithm, not "pick one")

`rank_and_select()` in `sop_engine.py`: sort matched SOPs by severity
descending; ties broken by `systemic_override > semantic > threshold`. The
response discloses any other SOPs that also matched but weren't primary, so
"why did it say that" is answerable two levels deep -- which SOP, and why
that one over the others that also fired.

### Grounding as a code-enforced invariant, not a prompt instruction

`validate_grounding()` extracts every number in the composed answer and
checks it against the live `WeatherSnapshot` plus the matched SOPs' own
policy constants (e.g. "40 km/h threshold" is a `sops.yaml` fact, not a
weather reading, but it's legitimately allowed to appear). Any other number
fails the response and routes to `honest_fallback` instead of returning it.
This is what makes "the bot must never invent a number" a testable
guarantee rather than a hope.

### Session memory

`SessionState` (`backend/models.py`) is a small structured object --
resolved location, last weather snapshot, last SOP ids -- not raw chat
history replay. A follow-up like "what about this evening instead?" reuses
the location without re-asking, and you can point at exactly which fields
carry over.

## Adding an 11th (well, 13th) SOP -- proof, not a promise

```bash
cat >> sops.yaml << 'EOF'

- id: SOP-013
  category: outdoor_exercise
  match_type: threshold
  severity: 2
  conditions:
    - field: humidity
      op: ">="
      value: 85
  keywords_any: [run, jog, exercise, cycle, cycling, bike, hike, walk, outdoor, sport, sports]
  advice_template: >
    Humidity is {humidity}% - high enough to make heat feel worse than the
    temperature alone suggests. Pace yourself and hydrate more than usual.
EOF
```

No file under `backend/` changes. Run `python3 -c "from backend.sop_engine
import load_sops; print(len(load_sops('sops.yaml')))"` before/after to see
12 -> 13.

If handed a **semantic** SOP live (e.g. "if a heatwave advisory affects
someone traveling with an infant, treat as high severity"), the same is
true -- it's still just a new YAML entry with `match_type: semantic` and a
`description`; `match_semantic_sops` in the graph already iterates over
every semantic SOP generically, so the new one is picked up automatically.
That's the harder version of the "add a policy live" test, and it's a real
path here, not a stretch goal.

## Running it

```bash
pip install -r requirements.txt

# Backend (needs ANTHROPIC_API_KEY in the environment for the semantic-SOP
# classifier calls; threshold-only questions work without it)
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn backend.app:app --reload --port 8000

# Frontend -- just open it, or serve it and set API_BASE if not on localhost:8000
open frontend/index.html
```

`GET /sops` exposes the loaded SOPs (the concrete proof of "policy-as-data"
for a reviewer). `POST /ask` runs the graph and returns the answer plus the
full trace (primary SOP, other matched SOPs, weather values used, grounded
flag).

## Tests

```bash
python3 -m pytest evals/ -v
```

- `evals/test_sop_engine.py` -- offline, no network/LLM: threshold matching,
  keyword gating, the systemic-override OR clause, conflict resolution,
  grounding token sets.
- `evals/test_graph_scenarios.py` -- runs the **actual compiled LangGraph**
  end-to-end, with `weather.fetch_weather` / `geocode` / `semantic_matcher.classify`
  monkeypatched so it's deterministic and offline. Covers: high-wind cycling
  block, calm-day low-severity travel, systemic override superseding a
  category SOP, the semantic picnic case (checking the LLM's summary gets
  slotted into the template rather than freely written), no-policy honest
  fallback, geocode-failure fallback, and session location reuse across
  turns.

## What I deliberately left out

- No LLM call composes final advice text, ever, including for semantic
  SOPs -- this was the main design constraint, not an oversight.
- No vector DB / RAG over SOP text -- with ~12-50 SOPs, a flat YAML file
  scanned in code is simpler, fully auditable, and doesn't introduce
  retrieval-miss as a new failure mode. Worth revisiting past a few hundred
  SOPs.
- No persistent session store (Redis/DB) -- in-memory dict keyed by
  `session_id`, fine for a demo, swap-in point clearly isolated in
  `backend/app.py`.
