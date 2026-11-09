"""Load football-data.co.uk CSVs (results, xG, closing-ish average odds) into plain records.

The snapshot in data/ is committed so tests and CI are reproducible; `python -m matchday.refresh`
pulls the current files.
"""
from __future__ import annotations

import csv
import difflib
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
SEASONS = ["2526", "2627"]  # oldest first; the last one is "current"


class UnknownName(ValueError):
    """A league or team the data doesn't have. The message carries the closest matches."""


@dataclass(frozen=True)
class Match:
    league: str
    season: str
    date: date
    home: str
    away: str
    hg: int | None = None  # None for a fixture not yet played
    ag: int | None = None
    hxg: float | None = None
    axg: float | None = None
    odds: tuple[float, float, float] | None = None  # market average, home/draw/away

    @property
    def result(self) -> str | None:
        if self.hg is None or self.ag is None:
            return None
        return "H" if self.hg > self.ag else "A" if self.hg < self.ag else "D"

    def to_dict(self) -> dict:
        d = {"date": self.date.isoformat(), "home": self.home, "away": self.away}
        if self.hg is not None:
            d |= {"score": f"{self.hg}-{self.ag}", "result": self.result}
        if self.hxg is not None:
            d |= {"xg": [self.hxg, self.axg]}
        return d


def _num(row: dict, key: str, cast=float):
    v = (row.get(key) or "").strip()
    try:
        return cast(v) if v else None
    except ValueError:
        return None


def _date(s: str) -> date:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"bad date {s!r}")


def _odds(row: dict) -> tuple[float, float, float] | None:
    for prefix in ("Avg", "B365", "PS"):
        o = tuple(_num(row, prefix + k) for k in "HDA")
        if all(x and x > 1 for x in o):
            return o  # type: ignore[return-value]
    return None


def read(path: Path, season: str) -> list[Match]:
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Div") not in LEAGUES or not row.get("HomeTeam"):
                continue
            out.append(Match(
                league=row["Div"], season=season, date=_date(row["Date"]),
                home=row["HomeTeam"].strip(), away=row["AwayTeam"].strip(),
                hg=_num(row, "FTHG", int), ag=_num(row, "FTAG", int),
                hxg=_num(row, "HxG"), axg=_num(row, "AxG"), odds=_odds(row),
            ))
    return out


@lru_cache(maxsize=1)
def results() -> tuple[Match, ...]:
    """Every played match in the snapshot, in date order."""
    ms = [m for lg in LEAGUES for s in SEASONS if (p := DATA / f"{lg}_{s}.csv").exists() for m in read(p, s)]
    return tuple(sorted((m for m in ms if m.hg is not None), key=lambda m: (m.date, m.league, m.home)))


@lru_cache(maxsize=1)
def fixtures() -> tuple[Match, ...]:
    """Upcoming fixtures with market odds, from fixtures.csv (unplayed rows only)."""
    p = DATA / "fixtures.csv"
    if not p.exists():
        return ()
    played = {(m.league, m.date, m.home, m.away) for m in results()}
    return tuple(m for m in read(p, SEASONS[-1]) if (m.league, m.date, m.home, m.away) not in played)


def league_code(name: str) -> str:
    """'E0', 'premier league', 'epl', 'la liga' ... -> the football-data.co.uk code."""
    key = _norm(name)
    aliases = {_norm(v): k for k, v in LEAGUES.items()} | {
        "epl": "E0", "england": "E0", "spain": "SP1", "laliga": "SP1", "germany": "D1",
        "italy": "I1", "seriea": "I1", "france": "F1", "ligue1": "F1",
    }
    if name.upper() in LEAGUES:
        return name.upper()
    if key.replace(" ", "") in aliases or key in aliases:
        return aliases.get(key) or aliases[key.replace(" ", "")]
    raise UnknownName(f"unknown league {name!r}; try one of {', '.join(f'{k} ({v})' for k, v in LEAGUES.items())}")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("manchester", "man").replace("'", "").replace(".", "")
    words = [w for w in s.split() if w not in {"fc", "afc", "cf", "ac", "sc", "the"}]
    return " ".join(words)


def teams(league: str, season: str | None = None) -> list[str]:
    season = season or SEASONS[-1]
    return sorted({t for m in results() if m.league == league and m.season == season for t in (m.home, m.away)})


def team_name(league: str, name: str) -> str:
    """Resolve a user's spelling ('Manchester United', 'spurs', 'bayern') to the dataset's name."""
    known = sorted({t for m in results() if m.league == league for t in (m.home, m.away)})
    by_norm = {_norm(t): t for t in known}
    q = _norm(name)
    if q in by_norm:
        return by_norm[q]
    alias = {"spurs": "tottenham", "wolves": "wolves", "barca": "barcelona", "inter milan": "inter",
             "psg": "paris sg", "bayern": "bayern munich", "atletico madrid": "ath madrid",
             "athletic bilbao": "ath bilbao", "real sociedad": "sociedad"}.get(q)
    if alias and alias in by_norm:
        return by_norm[alias]
    starts = [t for n, t in by_norm.items() if n.startswith(q) or q.startswith(n)]
    if len(starts) == 1:
        return starts[0]
    close = difflib.get_close_matches(q, list(by_norm), n=3, cutoff=0.6)
    if len(close) == 1 or (close and difflib.SequenceMatcher(None, q, close[0]).ratio() > 0.85):
        return by_norm[close[0]]
    hint = close and f"; did you mean {', '.join(by_norm[c] for c in close)}?" or ""
    raise UnknownName(f"no team {name!r} in {LEAGUES[league]}{hint}")
