"""The MCP server: football results, standings, a Poisson model and bookmaker odds as tools.

    python -m matchday.server                    # stdio, for Claude Desktop / Claude Code / any MCP client
    python -m matchday.server --http 8765        # streamable HTTP

Every tool returns structured output (a pydantic model), so an agent gets typed JSON, not prose to parse.
Bad names come back as tool errors that say what *is* available, so a model can correct itself.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from . import data
from .backtest import REPORT
from .data import LEAGUES, SEASONS, UnknownName
from .model import fit, grid, market, outcome

mcp = MCPServer(
    "matchday",
    instructions="Results, standings, form, a Poisson match model and bookmaker odds for the Premier League, "
                 "La Liga, Bundesliga, Serie A and Ligue 1 (football-data.co.uk). Leagues take a code (E0) or "
                 "a name; team names are matched loosely ('Manchester United' finds 'Man United').",
)


def _league(name: str) -> str:
    try:
        return data.league_code(name)
    except UnknownName as e:
        raise ToolError(str(e)) from e


def _team(lg: str, name: str) -> str:
    try:
        return data.team_name(lg, name)
    except UnknownName as e:
        raise ToolError(str(e)) from e


def _date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise ToolError(f"dates are YYYY-MM-DD, got {s!r}") from e


def _season(s: str | None) -> str:
    if s is None:
        return SEASONS[-1]
    s = s.replace("/", "").replace("-", "")
    s = ({f"20{x[:2]}20{x[2:]}": x for x in SEASONS} | {f"20{x}": x for x in SEASONS}).get(s, s)  # 2025-26, 2025/2026
    if s not in SEASONS:
        raise ToolError(f"seasons available: {', '.join(f'20{x[:2]}-{x[2:]}' for x in SEASONS)}")
    return s


class League(BaseModel):
    code: str
    name: str
    seasons: list[str]
    matches: int
    last_result: str


class Leagues(BaseModel):
    leagues: list[League]


@mcp.tool()
def list_leagues() -> Leagues:
    """The leagues and seasons loaded, with match counts and the date of the latest result."""
    rs = data.results()
    return Leagues(leagues=[
        League(code=k, name=v, seasons=[f"20{s[:2]}-{s[2:]}" for s in SEASONS],
               matches=sum(m.league == k for m in rs),
               last_result=max(m.date for m in rs if m.league == k).isoformat())
        for k, v in LEAGUES.items()
    ])


class Row(BaseModel):
    pos: int
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    gf: int
    ga: int
    gd: int
    points: int


class Table(BaseModel):
    league: str
    season: str
    asof: str
    rows: list[Row]


def table(lg: str, season: str, asof: date | None = None) -> list[Row]:
    acc: dict[str, list[int]] = {t: [0] * 6 for t in data.teams(lg, season)}  # P W D L GF GA
    for m in data.results():
        if m.league != lg or m.season != season or (asof and m.date > asof):
            continue
        for t, gf, ga in ((m.home, m.hg, m.ag), (m.away, m.ag, m.hg)):
            r = acc[t]
            r[0] += 1
            r[1 if gf > ga else 3 if gf < ga else 2] += 1
            r[4] += gf
            r[5] += ga
    ordered = sorted(acc.items(), key=lambda kv: (-(3 * kv[1][1] + kv[1][2]), -(kv[1][4] - kv[1][5]), -kv[1][4], kv[0]))
    return [Row(pos=i + 1, team=t, played=r[0], won=r[1], drawn=r[2], lost=r[3], gf=r[4], ga=r[5],
                gd=r[4] - r[5], points=3 * r[1] + r[2]) for i, (t, r) in enumerate(ordered)]


@mcp.tool()
def standings(league: str, season: str | None = None, asof: str | None = None) -> Table:
    """League table: points, then goal difference, then goals scored. `asof` (YYYY-MM-DD) rewinds it."""
    lg, s, d = _league(league), _season(season), _date(asof)
    rows = table(lg, s, d)
    last = max((m.date for m in data.results() if m.league == lg and m.season == s and (not d or m.date <= d)), default=None)
    return Table(league=LEAGUES[lg], season=f"20{s[:2]}-{s[2:]}", asof=(last or d or date.today()).isoformat(), rows=rows)


class Played(BaseModel):
    date: str
    home: str
    away: str
    score: str
    result: str
    xg: list[float] | None = None


class Form(BaseModel):
    team: str
    form: str = Field(description="oldest to newest, from this team's side: W/D/L")
    points_per_game: float
    goals_for: int
    goals_against: int
    matches: list[Played]


@mcp.tool()
def team_form(league: str, team: str, last: int = 5) -> Form:
    """A team's last N results (default 5), oldest first, with xG where the data has it."""
    lg = _league(league)
    t = _team(lg, team)
    ms = [m for m in data.results() if m.league == lg and t in (m.home, m.away)][-max(1, min(last, 38)):]
    letters, gf, ga = [], 0, 0
    for m in ms:
        f, a = (m.hg, m.ag) if m.home == t else (m.ag, m.hg)
        letters.append("W" if f > a else "L" if f < a else "D")
        gf, ga = gf + f, ga + a
    pts = sum({"W": 3, "D": 1, "L": 0}[x] for x in letters)
    return Form(team=t, form="".join(letters), points_per_game=round(pts / len(ms), 2), goals_for=gf,
                goals_against=ga, matches=[Played(**m.to_dict()) for m in ms])


class H2H(BaseModel):
    teams: list[str]
    meetings: list[Played]
    record: dict[str, int]


@mcp.tool()
def head_to_head(league: str, team_a: str, team_b: str) -> H2H:
    """Every meeting between two teams in the loaded seasons, and the W/D/W record."""
    lg = _league(league)
    a, b = _team(lg, team_a), _team(lg, team_b)
    ms = [m for m in data.results() if m.league == lg and {m.home, m.away} == {a, b}]
    rec = {a: 0, "draw": 0, b: 0}
    for m in ms:
        w = {"H": m.home, "A": m.away}.get(m.result, "draw")
        rec[w] += 1
    return H2H(teams=[a, b], meetings=[Played(**m.to_dict()) for m in ms], record=rec)


class Scoreline(BaseModel):
    score: str
    p: float


class Prediction(BaseModel):
    home: str
    away: str
    asof: str
    p_home: float
    p_draw: float
    p_away: float
    expected_goals: list[float]
    likely_scores: list[Scoreline]
    p_over_2_5: float
    p_both_score: float
    fitted_on: int = Field(description="matches the model was fitted on")


def predict(lg: str, home: str, away: str, asof: date | None = None) -> Prediction:
    asof = asof or max(m.date for m in data.results()) + timedelta(days=1)
    f = fit([m for m in data.results() if m.league == lg], asof)
    for t in (home, away):
        if t not in f.teams:
            raise ToolError(f"{t} has no matches before {asof}")
    lh, la = f.rates(home, away)
    g = grid(lh, la)
    ph, pd, pa = outcome(g)
    top = np.dstack(np.unravel_index(np.argsort(g, axis=None)[::-1][:5], g.shape))[0]
    goals = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
    return Prediction(
        home=home, away=away, asof=asof.isoformat(), p_home=round(ph, 3), p_draw=round(pd, 3), p_away=round(pa, 3),
        expected_goals=[round(lh, 2), round(la, 2)],
        likely_scores=[Scoreline(score=f"{i}-{j}", p=round(float(g[i, j] / g.sum()), 3)) for i, j in top],
        p_over_2_5=round(float(g[goals > 2.5].sum() / g.sum()), 3),
        p_both_score=round(float(g[1:, 1:].sum() / g.sum()), 3), fitted_on=f.n,
    )


@mcp.tool()
def predict_match(league: str, home: str, away: str, asof: str | None = None) -> Prediction:
    """Poisson forecast for home vs away: win/draw/loss, expected goals, likeliest scorelines, over 2.5,
    both teams to score. Fitted only on matches before `asof` (default: after the latest result)."""
    lg = _league(league)
    return predict(lg, _team(lg, home), _team(lg, away), _date(asof))


class Odds(BaseModel):
    date: str
    home: str
    away: str
    odds: list[float] = Field(description="market average decimal odds, home/draw/away")
    implied: list[float] = Field(description="probabilities with the bookmaker margin removed")
    margin: float
    score: str | None = None


@mcp.tool()
def market_odds(league: str, home: str, away: str) -> Odds:
    """Bookmaker average odds for the next fixture between these teams (or their latest meeting at this
    ground), converted to probabilities with the margin removed."""
    lg = _league(league)
    h, a = _team(lg, home), _team(lg, away)
    ms = [m for m in (*data.fixtures(), *reversed(data.results())) if m.league == lg and (m.home, m.away) == (h, a) and m.odds]
    if not ms:
        raise ToolError(f"no odds for {h} v {a} in the loaded data")
    m = ms[0]
    p = market(m.odds)
    return Odds(date=m.date.isoformat(), home=h, away=a, odds=list(m.odds), implied=[round(x, 3) for x in p],
                margin=round(sum(1 / o for o in m.odds) - 1, 4), score=m.to_dict().get("score"))


class Fixture(BaseModel):
    date: str
    home: str
    away: str
    odds: list[float] | None


class Fixtures(BaseModel):
    league: str
    fixtures: list[Fixture]
    note: str | None = None


@mcp.tool()
def upcoming_fixtures(league: str) -> Fixtures:
    """Scheduled fixtures with market odds, from the latest fixtures file."""
    lg = _league(league)
    fx = [Fixture(date=m.date.isoformat(), home=m.home, away=m.away, odds=list(m.odds) if m.odds else None)
          for m in data.fixtures() if m.league == lg]
    note = None if fx else "no unplayed fixtures in the snapshot; `python -m matchday.refresh` fetches the latest"
    return Fixtures(league=LEAGUES[lg], fixtures=fx, note=note)


class TrackRecord(BaseModel):
    goals: dict[str, dict[str, float]] = Field(description="all seasons: model vs market, league rates, uniform, always-home")
    xg_this_season: dict[str, dict[str, float]] = Field(description="this season only: goals vs xG vs blended fits, and the market")


@mcp.tool()
def model_track_record() -> TrackRecord:
    """How good the model is: the walk-forward backtest against bookmakers and naive baselines
    (log loss and Brier: lower is better; 'called' is the share of results it picked right)."""
    if not REPORT.exists():
        raise ToolError("no backtest report yet; run `python -m matchday.backtest`")
    return TrackRecord(**json.loads(REPORT.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, help="serve streamable HTTP on this port instead of stdio")
    args = ap.parse_args()
    if args.http:
        mcp.settings.port = args.http
        mcp.run("streamable-http")
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
