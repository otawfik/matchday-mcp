"""Walk-forward backtest: for every match day, fit on everything before it, predict that day.

    python -m matchday.backtest            # writes reports/backtest.json and prints the table
    python -m matchday.backtest --check    # also exit 1 unless the model beats every naive baseline

Compared against the bookmakers' average odds (margin removed), each league's historical H/D/A rates
and a uniform guess. Scored with log loss, Brier (sum over the three outcomes) and the share of
results called (highest-probability outcome).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from .data import LEAGUES, SEASONS, results
from .model import fit, grid, market, outcome

REPORT = Path(__file__).resolve().parent.parent / "reports" / "backtest.json"
WARMUP_DAYS = 42  # skip the first six weeks of the first season: nothing to fit yet


def score(rows: list[tuple[tuple[float, float, float], str]]) -> dict:
    idx = {"H": 0, "D": 1, "A": 2}
    ll = br = hit = 0.0
    for p, r in rows:
        k = idx[r]
        ll -= math.log(max(p[k], 1e-12))
        br += sum((p[i] - (i == k)) ** 2 for i in range(3))
        hit += max(range(3), key=lambda i: p[i]) == k
    n = len(rows)
    return {"n": n, "log_loss": round(ll / n, 4), "brier": round(br / n, 4), "called": round(hit / n, 4)}


def run(goals: str = "goals", seasons: list[str] | None = None) -> dict:
    seasons = seasons or SEASONS
    ms = list(results())
    start = min(m.date for m in ms) + timedelta(days=WARMUP_DAYS)
    rows: dict[str, list] = defaultdict(list)
    for lg in LEAGUES:
        lm = [m for m in ms if m.league == lg]
        for day in sorted({m.date for m in lm if m.date >= start and m.season in seasons}):
            past = [m for m in lm if m.date < day]
            f = fit(past, day, goals)
            base = [sum(m.result == r for m in past) / len(past) for r in "HDA"]
            for m in (m for m in lm if m.date == day):
                if m.home not in f.teams or m.away not in f.teams:
                    continue  # a team's first ever match in the data: no strength yet
                p = outcome(grid(*f.rates(m.home, m.away)))
                rows["model"].append((p, m.result))
                rows["market"].append((market(m.odds), m.result))
                rows["league_rates"].append((tuple(base), m.result))
                rows["uniform"].append(((1 / 3,) * 3, m.result))
                rows["always_home"].append(((1.0, 0.0, 0.0), m.result))
    out = {k: score(v) for k, v in rows.items() if k != "always_home"}
    out["always_home"] = {"called": score(rows["always_home"])["called"]}
    return out


def main() -> int:
    report = {g: run(g) for g in ("goals",)}
    report["xg_this_season"] = {g: run(g, [SEASONS[-1]])["model"] for g in ("goals", "xg", "blend")}
    report["xg_this_season"]["market"] = run("goals", [SEASONS[-1]])["market"]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if "--check" in sys.argv:
        r = report["goals"]
        bad = [f"{b} {k}: model {r['model'][k]} >= {r[b][k]}" for b in ("league_rates", "uniform")
               for k in ("log_loss", "brier") if r["model"][k] >= r[b][k]]
        if r["model"]["called"] <= r["always_home"]["called"]:
            bad.append("calls no more results than always picking the home side")
        if bad:
            print("MODEL GATE FAILED\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        print("model gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
