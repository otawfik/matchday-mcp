"""Three agents that write a matchday preview, coordinating through the MCP server.

    scout ─► analyst ─► editor:write ─► editor:check ─┬─► done
                            ▲                         │
                            └──── unsupported claims ─┘   (at most MAX_REVISIONS rewrites)

- **scout** talks to the MCP server (over stdio by default, the same way Claude Desktop does) and
  gathers a dossier per fixture: forecast, market odds, form, table, head-to-head. It passes `asof`
  on every call, so a replayed matchday never sees its own results.
- **analyst** turns dossiers into angles (model vs market disagreement, streaks, table stakes) and a
  ledger: every figure the preview is allowed to state, already formatted.
- **editor** writes the copy (a deterministic template; any `Writer`, such as an LLM, plugs in the
  same way) and then fact-checks it: every number in the text must appear in that fixture's ledger, and every team named
  must be playing in it. Failures go back to the writer with the list of problems; if a rewrite still
  fails, the offending sentences are cut rather than published.

    python -m matchday.agents --league E0                      # the next matchday
    python -m matchday.agents --league E0 --start 2026-11-07   # replay a past one
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

MAX_REVISIONS = 2
DISAGREE = 0.08  # model and market more than 8 points apart on some outcome: worth a line
NO_HISTORY = re.compile(r": (.+?) has no (?:matches|results) before")  # a promoted side's first game: no data, not a fault
NUMBER = re.compile(r"\d+-\d+|\d+(?:\.\d+)?(?:%|st|nd|rd|th)?")


class State(TypedDict, total=False):
    league: str
    start: str | None
    league_name: str
    fixtures: list[dict]
    dossiers: list[dict]
    calls: list[dict]
    angles: list[dict]
    drafts: dict[str, str]
    issues: dict[str, list[str]]
    revisions: int
    cut: list[str]
    preview: str


def server_params() -> StdioServerParameters | str:
    """Where the scout finds the MCP server: MATCHDAY_MCP_URL (streamable HTTP), else a stdio subprocess."""
    if url := os.environ.get("MATCHDAY_MCP_URL"):
        return url
    return StdioServerParameters(command=sys.executable, args=["-m", "matchday.server"])


# ── scout ─────────────────────────────────────────────────────────────────────────────────────────


class Tools:
    """An MCP session that logs every call (tool, arguments, latency, error) for the run report."""

    def __init__(self, client: Client, log: list[dict]):
        self.client, self.log = client, log

    async def __call__(self, name: str, **args: Any) -> dict | None:
        t = time.perf_counter()
        r = await self.client.call_tool(name, args)
        self.log.append({"tool": name, "args": args, "ms": round(1000 * (time.perf_counter() - t), 1),
                         "error": r.content[0].text if r.is_error else None})
        return None if r.is_error else r.structured_content


async def scout(state: State, tools: Tools) -> State:
    lg, start = state["league"], state.get("start")
    fx = await tools("upcoming_fixtures", league=lg, **({"start": start} if start else {}))
    dossiers = []
    for f in fx["fixtures"]:
        day = f["date"]
        # replay: everything is as of the morning of the match; live: as of now
        asof = {"asof": day} if start else {}
        d = {"fixture": f, "date": day}
        d["prediction"] = await tools("predict_match", league=lg, home=f["home"], away=f["away"], **asof)
        d["market"] = await tools("market_odds", league=lg, home=f["home"], away=f["away"], **({"on": day} if start else {}))
        d["form"] = [await tools("team_form", league=lg, team=t, last=5, **asof) for t in (f["home"], f["away"])]
        d["h2h"] = await tools("head_to_head", league=lg, team_a=f["home"], team_b=f["away"], **asof)
        dossiers.append(d)
    table = await tools("standings", league=lg, **({"asof": str(date.fromisoformat(start) - timedelta(days=1))} if start else {}))
    for d in dossiers:
        d["table"] = {r["team"]: r for r in table["rows"]} if table else {}
    return {"league_name": fx["league"], "fixtures": fx["fixtures"], "dossiers": dossiers}


# ── analyst ───────────────────────────────────────────────────────────────────────────────────────


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def streak(form: str) -> tuple[str, int]:
    """The run a team is on at the end of its form string: ('W', 4) for ...WWWW."""
    if not form:
        return "", 0
    n = len(form) - len(form.rstrip(form[-1]))
    return form[-1], n


def analyse(d: dict) -> dict:
    f, p, m = d["fixture"], d["prediction"], d["market"]
    home, away = f["home"], f["away"]
    facts: dict[str, str] = {}
    a: dict[str, Any] = {"key": f"{home} v {away}", "home": home, "away": away, "date": d["date"]}
    if p:
        probs = {"home": p["p_home"], "draw": p["p_draw"], "away": p["p_away"]}
        fav = max(probs, key=probs.get)
        a |= {"model": probs, "favourite": fav, "confidence": probs[fav]}
        facts |= {"p_home": pct(p["p_home"]), "p_draw": pct(p["p_draw"]), "p_away": pct(p["p_away"]),
                  "score": p["likely_scores"][0]["score"], "xg_home": f"{p['expected_goals'][0]:.2f}",
                  "xg_away": f"{p['expected_goals'][1]:.2f}", "over": pct(p["p_over_2_5"])}
    if p and m:
        mk = dict(zip(("home", "draw", "away"), m["implied"]))
        gaps = {k: a["model"][k] - mk[k] for k in mk}
        side = max(gaps, key=lambda k: abs(gaps[k]))
        a |= {"market": mk, "gap": gaps[side], "gap_side": side}
        facts |= {"m_home": pct(mk["home"]), "m_draw": pct(mk["draw"]), "m_away": pct(mk["away"]),
                  "gap": str(round(100 * abs(gaps[side])))}
    for side, team, fm in (("home", home, d["form"][0]), ("away", away, d["form"][1])):
        if fm:
            kind, n = streak(fm["form"])
            a[f"form_{side}"] = fm["form"]
            if n >= 3 and kind in "WL":
                a.setdefault("streaks", []).append((team, kind, n))
                facts[f"streak_{side}"] = str(n)
        row = d["table"].get(team)
        if row and row["played"]:
            a[f"pos_{side}"] = row["pos"]
            facts |= {f"pos_{side}": ordinal(row["pos"]), f"pts_{side}": str(row["points"])}
    if d["h2h"] and d["h2h"]["meetings"]:
        last = d["h2h"]["meetings"][-1]
        a["last_meeting"] = last
        facts |= {"h2h_score": last["score"], "h2h_year": last["date"][:4]}
    a["facts"] = facts
    # how much a reader should care: a real disagreement, a top-four clash, a long run
    a["interest"] = (abs(a.get("gap", 0)) * 4 + (a.get("pos_home", 20) <= 4 and a.get("pos_away", 20) <= 4)
                     + 0.1 * max((n for *_, n in a.get("streaks", [])), default=0))
    return a


async def analyst(state: State) -> State:
    angles = sorted((analyse(d) for d in state["dossiers"]), key=lambda a: -a["interest"])
    return {"angles": angles}


# ── editor ────────────────────────────────────────────────────────────────────────────────────────


def day_label(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%a} {d.day} {d:%b}"


def template(a: dict) -> str:
    """The deterministic writer: every figure comes straight from the ledger."""
    F, home, away = a["facts"], a["home"], a["away"]
    s = []
    if "pos_home" in F and "pos_away" in F:
        s.append(f"{home} ({F['pos_home']}, {F['pts_home']} pts) host {away} ({F['pos_away']}, {F['pts_away']} pts).")
    for team, kind, n in a.get("streaks", []):
        s.append(f"{team} have {'won' if kind == 'W' else 'lost'} {n} in a row.")
    if "form_home" in a and "form_away" in a:
        s.append(f"Form, oldest to newest: {home} {a['form_home']}, {away} {a['form_away']}.")
    if "model" in a:
        fav = a["favourite"]
        if fav == "draw":
            s.append(f"The model leans to a draw ({F['p_draw']}; {home} {F['p_home']}, {away} {F['p_away']}).")
        else:
            team, other = (home, away) if fav == "home" else (away, home)
            s.append(f"The model makes {team} favourites at {F['p_' + fav]} (draw {F['p_draw']}, "
                     f"{other} {F['p_away' if fav == 'home' else 'p_home']}), and {F['score']} is the likeliest score.")
        s.append(f"Expected goals {F['xg_home']} to {F['xg_away']}; {F['over']} for over 2.5.")
    if "market" in a:
        side = a["gap_side"]
        label = {"home": home, "draw": "the draw", "away": away}[side]
        if abs(a["gap"]) >= DISAGREE:
            more = "more" if a["gap"] > 0 else "less"
            s.append(f"The bookmakers disagree: the model rates {label} {F['gap']} points {more} likely than the "
                     f"market does ({F['p_' + side]} against {F['m_' + side]}).")
        else:
            s.append(f"The bookmakers broadly agree ({home} {F['m_home']}, draw {F['m_draw']}, {away} {F['m_away']}).")
    if "last_meeting" in a:
        lm = a["last_meeting"]
        s.append(f"Last meeting: {lm['home']} {F['h2h_score']} {lm['away']} ({F['h2h_year']}).")
    return " ".join(s)


def intro(state: State) -> tuple[str, set[str]]:
    angles = state["angles"]
    days = sorted({a["date"] for a in angles})
    lines = [f"## {state['league_name']} preview: {day_label(days[0])} to {day_label(days[-1])}",
             f"{len(angles)} fixtures."]
    allowed = {str(len(angles)), *(str(date.fromisoformat(d).day) for d in days)}
    sure = max((a for a in angles if "model" in a and a["favourite"] != "draw"), key=lambda a: a["confidence"], default=None)
    if sure:
        team = sure["home"] if sure["favourite"] == "home" else sure["away"]
        lines.append(f"Most confident call: {team} ({pct(sure['confidence'])}).")
        allowed.add(pct(sure["confidence"]))
    split = max((a for a in angles if "gap" in a), key=lambda a: abs(a["gap"]), default=None)
    if split and abs(split["gap"]) >= DISAGREE:
        lines.append(f"Biggest split with the bookmakers: {split['key']} ({round(100 * abs(split['gap']))} points).")
        allowed.add(str(round(100 * abs(split["gap"]))))
    return lines[0] + "\n\n" + " ".join(lines[1:]), allowed


def figures(text: str, names: set[str]) -> list[str]:
    """The numbers a text states, not counting digits inside team names (Schalke 04 isn't a figure)."""
    for n in sorted(names, key=len, reverse=True):
        text = text.replace(n, " ")
    return NUMBER.findall(text)


def check(text: str, allowed: set[str], teams: set[str], league_teams: set[str]) -> list[str]:
    """Every number must be in the ledger; every team named must be one of this fixture's."""
    problems = [f"unsupported figure {tok!r}" for tok in figures(text, league_teams | teams) if tok not in allowed]
    problems += [f"names {t}, who isn't in this fixture" for t in sorted(league_teams - teams)
                 if re.search(rf"(?<![\w']){re.escape(t)}(?![\w'])", text)]
    return problems


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())


Writer = Callable[[dict, list[str]], str]


def make_editor(writer: Writer, league_teams: set[str]):
    async def write(state: State) -> State:
        drafts = dict(state.get("drafts") or {})
        issues = state.get("issues") or {}
        for a in state["angles"]:
            if a["key"] not in drafts or issues.get(a["key"]):
                drafts[a["key"]] = writer(a, issues.get(a["key"], []))
        return {"drafts": drafts, "revisions": state.get("revisions", -1) + 1}

    async def check_all(state: State) -> State:
        issues = {}
        for a in state["angles"]:
            allowed = {*a["facts"].values(), *(str(date.fromisoformat(a["date"]).day),), "2.5"}
            if bad := check(state["drafts"][a["key"]], allowed, {a["home"], a["away"]}, league_teams):
                issues[a["key"]] = bad
        return {"issues": issues}

    async def publish(state: State) -> State:
        head, _ = intro(state)
        parts, cut = [head], []
        for a in state["angles"]:
            text = state["drafts"][a["key"]]
            if state["issues"].get(a["key"]):  # out of rewrites: cut what can't be backed up
                allowed = {*a["facts"].values(), str(date.fromisoformat(a["date"]).day), "2.5"}
                keep = []
                for s in sentences(text):
                    (cut if check(s, allowed, {a["home"], a["away"]}, league_teams) else keep).append(s)
                text = " ".join(keep)
            parts.append(f"### {a['key']} · {day_label(a['date'])}\n\n{text}")
        return {"preview": "\n\n".join(parts) + "\n", "cut": cut}

    return write, check_all, publish


def route(state: State) -> str:
    return "write" if state["issues"] and state["revisions"] < MAX_REVISIONS else "publish"


# ── the graph ─────────────────────────────────────────────────────────────────────────────────────


def template_writer(a: dict, _issues: list[str]) -> str:
    return template(a)


async def run(league: str, start: str | None = None, server: Any = None, writer: Writer | None = None) -> dict:
    """One preview run. `server` is anything mcp.client.Client accepts (default: stdio subprocess)."""
    t0 = time.perf_counter()
    calls: list[dict] = []
    async with Client(server if server is not None else server_params()) as client:
        tools = Tools(client, calls)
        table = await tools("standings", league=league)
        league_teams = {r["team"] for r in table["rows"]} if table else set()
        write, check_all, publish = make_editor(writer or template_writer, league_teams)

        async def scout_node(s: State) -> State:
            return await scout(s, tools)

        g = StateGraph(State)
        g.add_node("scout", scout_node)
        g.add_node("analyst", analyst)
        g.add_node("write", write)
        g.add_node("check", check_all)
        g.add_node("publish", publish)
        g.set_entry_point("scout")
        g.add_edge("scout", "analyst")
        g.add_edge("analyst", "write")
        g.add_edge("write", "check")
        g.add_conditional_edges("check", route, {"write": "write", "publish": "publish"})
        g.add_edge("publish", END)
        out = await g.compile().ainvoke({"league": league, "start": start, "calls": calls})
    return {
        "league": league, "start": start, "preview": out["preview"], "fixtures": len(out["angles"]),
        "angles": [{k: v for k, v in a.items() if k != "facts"} | {"facts": a["facts"]} for a in out["angles"]],
        "revisions": out["revisions"], "cut": out["cut"], "tool_calls": len(calls),
        "tool_errors": [c for c in calls if c["error"] and not NO_HISTORY.search(c["error"])],
        "no_history": sorted({m.group(1) for c in calls if c["error"] and (m := NO_HISTORY.search(c["error"]))}),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--start", help="replay the matchday starting on this date (YYYY-MM-DD)")
    ap.add_argument("--json", action="store_true", help="print the whole run report, not just the preview")
    args = ap.parse_args()
    r = asyncio.run(run(args.league, args.start))
    print(json.dumps(r, indent=2, default=str) if args.json else r["preview"])


if __name__ == "__main__":
    main()
