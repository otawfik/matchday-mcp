"""Webhook trigger: POST a signed event, get a run id back at once, and the agents run in the background.

    MATCHDAY_WEBHOOK_SECRET=... uvicorn matchday.webhook:app --port 8000

    POST /webhooks/matchday   {"league": "E0"}  or  {"league": "E0", "start": "2026-11-07"}  (replay)
        X-Matchday-Signature: sha256=<HMAC-SHA256 of the raw body with the secret>
        X-Matchday-Delivery:  <unique id>   a redelivery with the same id returns the original run
    GET  /runs/{id}           status, then the preview and the run report
    GET  /health

Signed like GitHub's webhooks, so a GitHub Action, a cron job or any service that can sign a body can
fire it. Unsigned or wrongly signed requests get 401 before anything is parsed, and with no secret
configured the endpoint refuses everything rather than running open.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import uuid
from datetime import date
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, field_validator

from . import agents
from .data import LEAGUES, league_code

app = FastAPI(title="matchday agents")
RUNS: dict[str, dict] = {}
DELIVERIES: dict[str, str] = {}
SERVER: Any = None  # what the agents connect to; None = stdio subprocess. Tests point it at the in-process server.


class Event(BaseModel):
    league: str
    start: str | None = None

    @field_validator("league")
    @classmethod
    def known_league(cls, v: str) -> str:
        return league_code(v)

    @field_validator("start")
    @classmethod
    def iso_date(cls, v: str | None) -> str | None:
        return v and date.fromisoformat(v).isoformat()


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def execute(run_id: str, ev: Event) -> None:
    RUNS[run_id]["status"] = "running"
    try:
        RUNS[run_id] |= {"status": "done", "result": await agents.run(ev.league, ev.start, server=SERVER)}
    except Exception as e:  # a failed run is reported on the run, not swallowed
        RUNS[run_id] |= {"status": "failed", "error": f"{type(e).__name__}: {e}"}


@app.post("/webhooks/matchday", status_code=202)
async def matchday(request: Request, x_matchday_signature: str = Header(""), x_matchday_delivery: str = Header("")):
    secret = os.environ.get("MATCHDAY_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(503, "MATCHDAY_WEBHOOK_SECRET is not set; refusing unsigned triggers")
    body = await request.body()
    if not hmac.compare_digest(sign(body, secret), x_matchday_signature):
        raise HTTPException(401, "bad signature")
    if x_matchday_delivery and x_matchday_delivery in DELIVERIES:
        rid = DELIVERIES[x_matchday_delivery]
        return {"run_id": rid, "status": RUNS[rid]["status"], "duplicate": True}
    try:
        ev = Event(**json.loads(body))
    except Exception as e:
        raise HTTPException(422, f"bad event: {e}; leagues: {', '.join(LEAGUES)}") from e
    rid = uuid.uuid4().hex[:12]
    RUNS[rid] = {"status": "queued", "event": ev.model_dump()}
    if x_matchday_delivery:
        DELIVERIES[x_matchday_delivery] = rid
    RUNS[rid]["task"] = asyncio.create_task(execute(rid, ev))
    return {"run_id": rid, "status": "queued"}


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "no such run")
    return {k: v for k, v in RUNS[run_id].items() if k != "task"}


@app.get("/health")
async def health():
    return {"ok": True, "runs": len(RUNS)}
