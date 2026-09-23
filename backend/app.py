"""FastAPI entrypoint.

POST /ask {"session_id": "...", "question": "...", "location_hint": "..."}
  -> runs the graph, returns a BotResponse (as JSON) with full traceability:
     which SOP fired, which others also matched, and the exact weather
     values used.

Session state is kept in-memory per session_id (fine for a take-home demo;
swap for Redis/DB in production).
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graph import build_graph
from .models import SessionState
from .sop_engine import load_sops

SOPS_PATH = Path(__file__).resolve().parent.parent / "sops.yaml"

app = FastAPI(title="Weather-Advisory Support Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()
_sops = load_sops(SOPS_PATH)
_sessions: dict[str, SessionState] = {}


class AskRequest(BaseModel):
    session_id: str
    question: str
    location_hint: str | None = None


class AskResponse(BaseModel):
    answer: str
    primary_sop_id: str | None
    other_matched_sop_ids: list[str]
    weather_snapshot: dict | None
    grounded: bool
    fallback_reason: str | None
    location_name: str | None


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    session = _sessions.setdefault(req.session_id, SessionState())

    result_state = await _graph.ainvoke(
        {
            "question": req.question,
            "location_hint": req.location_hint,
            "session": session,
            "sops": _sops,
        }
    )
    resp = result_state["response"]
    return AskResponse(
        answer=resp.answer,
        primary_sop_id=resp.primary_sop_id,
        other_matched_sop_ids=resp.other_matched_sop_ids,
        weather_snapshot=resp.weather_snapshot,
        grounded=resp.grounded,
        fallback_reason=resp.fallback_reason,
        location_name=session.location_name,
    )


@app.get("/sops")
async def list_sops() -> list[dict]:
    """Expose the loaded SOPs so the frontend / a reviewer can show
    'policy-as-data' concretely -- add SOP-013 to sops.yaml and it shows up
    here with zero code changes."""
    return [
        {
            "id": s.id,
            "category": s.category,
            "match_type": s.match_type,
            "severity": s.severity,
            "description": s.description,
        }
        for s in _sops
    ]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "sops_loaded": len(_sops)}


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
