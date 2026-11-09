"""Every tool, called through a real MCP client session against the in-process server."""
import anyio
import pytest
from mcp.client import Client

from matchday.server import mcp, table


def call(name: str, args: dict | None = None):
    async def go():
        async with Client(mcp) as c:
            return await c.call_tool(name, args or {})
    return anyio.run(go)


def ok(name: str, args: dict | None = None) -> dict:
    r = call(name, args)
    assert not r.is_error, r.content[0].text
    return r.structured_content


def test_tools_are_listed_with_schemas():
    async def go():
        async with Client(mcp) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"list_leagues", "standings", "team_form", "head_to_head", "predict_match",
                          "market_odds", "upcoming_fixtures", "model_track_record"}
    assert all(t.description for t in tools.values())
    assert tools["predict_match"].input_schema["required"] == ["league", "home", "away"]


def test_list_leagues():
    ls = ok("list_leagues")["leagues"]
    assert [x["code"] for x in ls] == ["E0", "SP1", "D1", "I1", "F1"]
    assert all(x["matches"] > 300 for x in ls)


def test_standings_are_consistent():
    t = ok("standings", {"league": "premier league", "season": "2025-26"})
    rows = t["rows"]
    assert len(rows) == 20 and all(r["played"] == 38 for r in rows)
    assert sum(r["gf"] for r in rows) == sum(r["ga"] for r in rows)
    assert all(r["points"] == 3 * r["won"] + r["drawn"] for r in rows)
    assert [r["points"] for r in rows] == sorted((r["points"] for r in rows), reverse=True)


def test_standings_asof_rewinds():
    early = ok("standings", {"league": "E0", "asof": "2026-08-31"})
    assert max(r["played"] for r in early["rows"]) <= 4


def test_team_form_from_the_teams_side():
    f = ok("team_form", {"league": "E0", "team": "Manchester United", "last": 5})
    assert f["team"] == "Man United" and len(f["form"]) == 5 == len(f["matches"])
    assert set(f["form"]) <= set("WDL")


def test_head_to_head_record_adds_up():
    h = ok("head_to_head", {"league": "D1", "team_a": "bayern", "team_b": "Dortmund"})
    assert sum(h["record"].values()) == len(h["meetings"]) >= 2


def test_predict_match():
    p = ok("predict_match", {"league": "La Liga", "home": "Barca", "away": "Real Madrid"})
    assert p["p_home"] + p["p_draw"] + p["p_away"] == pytest.approx(1, abs=0.002)
    assert len(p["likely_scores"]) == 5 and 0 < p["p_over_2_5"] < 1
    assert p["fitted_on"] > 400


def test_predict_is_walk_forward():
    early = ok("predict_match", {"league": "E0", "home": "Arsenal", "away": "Chelsea", "asof": "2025-09-01"})
    assert early["fitted_on"] < 40


def test_market_odds():
    o = ok("market_odds", {"league": "E0", "home": "Man United", "away": "Aston Villa"})
    assert sum(o["implied"]) == pytest.approx(1, abs=0.002) and o["margin"] > 0


def test_track_record_beats_baselines():
    r = ok("model_track_record")["goals"]
    assert r["model"]["log_loss"] < r["league_rates"]["log_loss"] < r["uniform"]["log_loss"]


@pytest.mark.parametrize("name,args,msg", [
    ("standings", {"league": "MLS"}, "unknown league"),
    ("team_form", {"league": "E0", "team": "Man Cty Utd"}, "did you mean"),
    ("standings", {"league": "E0", "season": "1999"}, "seasons available"),
    ("standings", {"league": "E0", "asof": "yesterday"}, "YYYY-MM-DD"),
])
def test_bad_input_is_a_readable_tool_error(name, args, msg):
    r = call(name, args)
    assert r.is_error and msg in r.content[0].text


def test_table_helper_empty_before_season():
    from datetime import date
    rows = table("E0", "2627", date(2026, 8, 1))
    assert all(r.played == 0 for r in rows)


def test_stdio_subprocess_end_to_end():
    """What Claude Desktop / Claude Code actually do: spawn `python -m matchday.server` and talk JSON-RPC over stdio."""
    import sys
    from pathlib import Path

    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "matchday.server"],
                                   cwd=str(Path(__file__).resolve().parent.parent))

    async def go():
        async with Client(params) as c:
            names = [t.name for t in (await c.list_tools()).tools]
            r = await c.call_tool("predict_match", {"league": "E0", "home": "Arsenal", "away": "Man City"})
            return names, r

    names, r = anyio.run(go)
    assert "predict_match" in names and not r.is_error
    assert r.structured_content["home"] == "Arsenal"
