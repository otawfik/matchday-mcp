"""A Maher-style Poisson model: each team has an attack and a defence strength, plus a home factor.

Fitted by weighted maximum likelihood (the usual fixed-point updates) on every match before the
prediction date, with exponential time decay so this season counts more than last, and a pseudo-count
prior that pulls teams with few matches (the promoted ones in August) towards league average.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from .data import Match

HALF_LIFE_DAYS = 120
PRIOR_MATCHES = 4.0
MAX_GOALS = 10


@dataclass
class Fit:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray  # > 1 concedes more than average
    home: float
    mu: float  # league goals per team per match
    n: int

    def rates(self, home: str, away: str) -> tuple[float, float]:
        i, j = self.teams.index(home), self.teams.index(away)
        return (self.mu * self.home * self.attack[i] * self.defence[j],
                self.mu * self.attack[j] * self.defence[i])


def fit(matches: list[Match], asof: date, goals: str = "goals") -> Fit:
    """goals='goals' fits real scorelines; 'xg' fits expected goals; 'blend' averages the two."""
    ms = [m for m in matches if m.date < asof and m.hg is not None]
    if not ms:
        raise ValueError("no matches before that date")
    teams = sorted({t for m in ms for t in (m.home, m.away)})
    ix = {t: k for k, t in enumerate(teams)}
    h = np.array([ix[m.home] for m in ms])
    a = np.array([ix[m.away] for m in ms])
    gh, ga = _goals(ms, goals)
    w = np.array([0.5 ** ((asof - m.date).days / HALF_LIFE_DAYS) for m in ms])

    mu = (w @ (gh + ga)) / (2 * w.sum())
    home = (w @ gh) / max(w @ ga, 1e-9)
    att = np.ones(len(teams))
    dfn = np.ones(len(teams))
    k = PRIOR_MATCHES * mu  # prior: k goals over k expected goals, i.e. strength 1
    for _ in range(60):
        # attack: goals scored / goals expected against those defences
        sc = np.bincount(h, w * gh, len(teams)) + np.bincount(a, w * ga, len(teams))
        ex = np.bincount(h, w * mu * home * dfn[a], len(teams)) + np.bincount(a, w * mu * dfn[h], len(teams))
        att = (sc + k) / (ex + k)
        cn = np.bincount(a, w * gh, len(teams)) + np.bincount(h, w * ga, len(teams))
        ex = np.bincount(a, w * mu * home * att[h], len(teams)) + np.bincount(h, w * mu * att[a], len(teams))
        dfn = (cn + k) / (ex + k)
        home = (w @ gh) / (w @ (mu * att[h] * dfn[a]))
    return Fit(teams, att, dfn, float(home), float(mu), len(ms))


def _goals(ms: list[Match], kind: str) -> tuple[np.ndarray, np.ndarray]:
    gh = np.array([m.hg for m in ms], float)
    ga = np.array([m.ag for m in ms], float)
    if kind == "goals":
        return gh, ga
    xh = np.array([m.hxg if m.hxg is not None else m.hg for m in ms], float)
    xa = np.array([m.axg if m.axg is not None else m.ag for m in ms], float)
    if kind == "xg":
        return xh, xa
    return (gh + xh) / 2, (ga + xa) / 2


def grid(lh: float, la: float) -> np.ndarray:
    """P(home scores i, away scores j) for i, j up to MAX_GOALS."""
    k = np.arange(MAX_GOALS + 1)
    fact = np.array([math.factorial(x) for x in k], float)
    ph = np.exp(-lh) * lh ** k / fact
    pa = np.exp(-la) * la ** k / fact
    return np.outer(ph, pa)


def outcome(g: np.ndarray) -> tuple[float, float, float]:
    s = g.sum()
    return float(np.tril(g, -1).sum() / s), float(np.trace(g) / s), float(np.triu(g, 1).sum() / s)


def market(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    """Bookmaker odds -> probabilities, margin removed proportionally."""
    inv = [1 / o for o in odds]
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s
