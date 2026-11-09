"""Evaluate the agents by replaying every matchday of the current season in all five leagues.

    python -m matchday.evaluate            # writes reports/agents.json
    python -m matchday.evaluate --check    # also exit 1 if coverage or grounding drop below 100%,
                                           # or the fact-check misses more than 5% of planted errors

- **coverage**: fixtures previewed / fixtures played in the window
- **grounding**: numbers in the published previews that trace to a tool output (re-checked from outside)
- **mutation test**: in every published section, change one figure or swap in a team from another
  match, and see whether the editor's fact-check catches it
- **calls**: how the confident calls and the model-vs-market arguments turned out against real results
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

from .agents import NUMBER, check, figures, run
from .data import LEAGUES, SEASONS, results
from .server import mcp

REPORT = Path(__file__).resolve().parent.parent / "reports" / "agents.json"
FIRST = date(2026, 8, 29)  # skip the opening fortnight: no current-season form or table yet
WINDOW = 4


def windows(league: str) -> list[str]:
    days = sorted({m.date for m in results() if m.league == league and m.season == SEASONS[-1] and m.date >= FIRST})
    out, end = [], None
    for d in days:
        if end is None or d >= end:
            out.append(d.isoformat())
            end = d + timedelta(days=WINDOW)
    return out


def mutate(text: str, facts: set[str], others: list[str], rng: random.Random) -> list[str]:
    """One planted error per kind: a figure nudged to a value not in the ledger, and a stranger's name."""
    out = []
    toks = [m for m in NUMBER.finditer(text) if m.group() in facts and not re.search(r"[A-Za-z] $", text[: m.start()][-2:])]
    if toks:
        m = rng.choice(toks)
        tok = m.group()
        for delta in (7, 11, 13, 3):
            if tok.endswith("%"):
                new = f"{min(99, int(tok[:-1]) + delta)}%"
            elif "-" in tok:
                a, b = tok.split("-")
                new = f"{int(a) + delta % 4 + 1}-{b}"
            elif tok[-2:] in ("st", "nd", "rd", "th"):
                n = int(tok[:-2]) + delta % 5 + 1
                new = f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
            elif "." in tok:
                new = f"{float(tok) + delta / 10:.2f}"
            else:
                new = str(int(tok) + delta)
            if new != tok:
                break
        out.append(text[: m.start()] + new + text[m.end():])
    if others:
        out.append(text + f" {rng.choice(others)} will be watching.")
    return out


def outcome_of(league: str, home: str, away: str, day: str) -> str:
    return next(m.result for m in results() if m.league == league and m.home == home and m.away == away
                and m.date.isoformat() == day)


async def evaluate() -> dict:
    rng = random.Random(7)
    runs, secs = [], []
    planted = caught = 0
    planted_kind = {"figure": [0, 0], "team": [0, 0]}
    numbers = grounded = played = covered = 0
    confident = []  # (prob, won)
    splits = []  # (model p of outcome, market p of outcome) when they disagreed
    for lg in LEAGUES:
        lteams = sorted({t for m in results() if m.league == lg and m.season == SEASONS[-1] for t in (m.home, m.away)})
        for start in windows(lg):
            r = await run(lg, start, server=mcp)
            secs.append(r["seconds"])
            end = date.fromisoformat(start) + timedelta(days=WINDOW)
            played += sum(m.league == lg and date.fromisoformat(start) <= m.date < end for m in results())
            covered += r["fixtures"]
            sections = dict(s.split("\n\n", 1) for s in r["preview"].split("\n### ")[1:])
            for a in r["angles"]:
                body = next(v for k, v in sections.items() if k.startswith(a["key"]))
                allowed = {*a["facts"].values(), str(int(a["date"][8:])), "2.5"}
                toks = figures(body, set(lteams))
                numbers += len(toks)
                grounded += sum(t in allowed for t in toks)
                teams = {a["home"], a["away"]}
                others = [t for t in lteams if t not in teams]
                for kind, bad in zip(("figure", "team"), mutate(body, allowed, others, rng)):
                    hit = bool(check(bad, allowed, teams, set(lteams)))
                    planted += 1
                    caught += hit
                    planted_kind[kind][0] += 1
                    planted_kind[kind][1] += hit
                res = outcome_of(lg, a["home"], a["away"], a["date"])
                k = {"H": "home", "D": "draw", "A": "away"}[res]
                if "model" in a and a["favourite"] != "draw" and a["confidence"] >= 0.6:
                    confident.append((a["confidence"], a["favourite"] == k))
                if "gap" in a and abs(a["gap"]) >= 0.08:
                    splits.append((a["model"][k], a["market"][k]))
            runs.append({"league": lg, "start": start, "fixtures": r["fixtures"], "revisions": r["revisions"],
                         "cut": len(r["cut"]), "tool_calls": r["tool_calls"], "tool_errors": len(r["tool_errors"]),
                         "no_history": r["no_history"]})

    def ll(ps: list[float]) -> float:
        return round(-sum(math.log(max(p, 1e-12)) for p in ps) / len(ps), 4)

    return {
        "runs": len(runs), "fixtures": covered, "coverage": round(covered / played, 4),
        "grounding": round(grounded / numbers, 4), "numbers_checked": numbers,
        "mutation": {"planted": planted, "caught": round(caught / planted, 4),
                     **{k: {"planted": v[0], "caught": round(v[1] / v[0], 4)} for k, v in planted_kind.items()}},
        "tool_calls_per_fixture": round(sum(r["tool_calls"] for r in runs) / covered, 2),
        "tool_errors": sum(r["tool_errors"] for r in runs),
        "revisions": sum(r["revisions"] for r in runs),
        "sentences_cut": sum(r["cut"] for r in runs),
        "no_history": sorted({t for r in runs for t in r["no_history"]}),
        "run_seconds_p50": round(statistics.median(secs), 2),
        "confident_calls": {"n": len(confident), "mean_confidence": round(statistics.mean(p for p, _ in confident), 3),
                            "won": round(sum(w for _, w in confident) / len(confident), 3)},
        "model_vs_market_splits": {"n": len(splits), "model_log_loss": ll([m for m, _ in splits]),
                                   "market_log_loss": ll([k for _, k in splits]),
                                   "model_closer": round(sum(m > k for m, k in splits) / len(splits), 3)},
        "per_run": runs,
    }


def main() -> int:
    rep = asyncio.run(evaluate())
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_run"}, indent=2))
    if "--check" in sys.argv:
        bad = [k for k, ok in (("coverage", rep["coverage"] == 1), ("grounding", rep["grounding"] == 1),
                               ("mutation", rep["mutation"]["caught"] >= 0.95), ("tool_errors", rep["tool_errors"] == 0),
                               ("template rewrites", rep["revisions"] == 0)) if not ok]
        if bad:
            print("AGENT GATE FAILED: " + ", ".join(bad), file=sys.stderr)
            return 1
        print("agent gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
