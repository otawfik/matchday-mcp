"""The webhook: signatures, validation, idempotent redelivery, and a run going all the way to a preview."""
import asyncio
import json

import httpx
import pytest

from matchday import webhook
from matchday.server import mcp

SECRET = "test-secret"


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setenv("MATCHDAY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(webhook, "SERVER", mcp)
    webhook.RUNS.clear()
    webhook.DELIVERIES.clear()


def post(c, event: dict, secret=SECRET, delivery=""):
    body = json.dumps(event).encode()
    return c.post("/webhooks/matchday", content=body, headers={
        "X-Matchday-Signature": webhook.sign(body, secret), "X-Matchday-Delivery": delivery,
        "content-type": "application/json"})


def session(fn):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=webhook.app), base_url="http://t") as c:
            return await fn(c)
    return asyncio.run(go())


def test_refuses_everything_without_a_secret(monkeypatch):
    monkeypatch.delenv("MATCHDAY_WEBHOOK_SECRET")
    r = session(lambda c: post(c, {"league": "E0"}))
    assert r.status_code == 503


def test_rejects_a_bad_signature():
    r = session(lambda c: post(c, {"league": "E0"}, secret="wrong"))
    assert r.status_code == 401 and webhook.RUNS == {}


def test_rejects_an_unknown_league():
    r = session(lambda c: post(c, {"league": "MLS"}))
    assert r.status_code == 422 and "E0" in r.json()["detail"]


def test_signed_event_runs_the_agents_to_a_preview():
    async def flow(c):
        r = await post(c, {"league": "premier league", "start": "2026-11-07"}, delivery="d-1")
        assert r.status_code == 202
        rid = r.json()["run_id"]
        again = await post(c, {"league": "premier league", "start": "2026-11-07"}, delivery="d-1")
        assert again.json() | {"status": None} == {"run_id": rid, "status": None, "duplicate": True}
        await webhook.RUNS[rid]["task"]
        return (await c.get(f"/runs/{rid}")).json()

    run = session(flow)
    assert run["status"] == "done", run
    assert run["event"] == {"league": "E0", "start": "2026-11-07"}
    assert run["result"]["fixtures"] == 9 and run["result"]["preview"].startswith("## Premier League preview")
    assert len(webhook.RUNS) == 1


def test_unknown_run_is_404():
    assert session(lambda c: c.get("/runs/nope")).status_code == 404
