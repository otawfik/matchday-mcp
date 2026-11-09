"""Data loading, name resolution, the model's maths and the backtest scorer. Fast: no MCP here."""
from datetime import date

import numpy as np
import pytest

from matchday import data
from matchday.backtest import score
from matchday.data import Match, UnknownName
from matchday.model import fit, grid, market, outcome


def test_snapshot_loads_every_league_and_season():
    rs = data.results()
    assert {(m.league, m.season) for m in rs} == {(lg, s) for lg in data.LEAGUES for s in data.SEASONS}
    assert all(m.odds and all(o > 1 for o in m.odds) for m in rs)
    assert list(rs) == sorted(rs, key=lambda m: (m.date, m.league, m.home))


@pytest.mark.parametrize("q,lg,want", [
    ("Manchester United", "E0", "Man United"), ("spurs", "E0", "Tottenham"), ("arsnal", "E0", "Arsenal"),
    ("Liverpool FC", "E0", "Liverpool"), ("bayern", "D1", "Bayern Munich"), ("Barca", "SP1", "Barcelona"),
])
def test_team_names_resolve(q, lg, want):
    assert data.team_name(lg, q) == want


def test_unknown_team_suggests_close_names():
    with pytest.raises(UnknownName, match="did you mean.*Man City"):
        data.team_name("E0", "Man Cty Utd")


@pytest.mark.parametrize("q,code", [("E0", "E0"), ("premier league", "E0"), ("La Liga", "SP1"), ("serie a", "I1"), ("epl", "E0")])
def test_league_codes(q, code):
    assert data.league_code(q) == code


def test_grid_is_a_distribution_and_outcomes_sum_to_one():
    g = grid(1.6, 1.1)
    assert g.sum() == pytest.approx(1, abs=1e-5)
    assert sum(outcome(g)) == pytest.approx(1)
    ph, _, pa = outcome(g)
    assert ph > pa


def test_market_removes_the_margin():
    p = market((2.0, 3.4, 3.8))
    assert sum(p) == pytest.approx(1)
    assert p[0] > p[2] > 0


def _synthetic(strong="A", weak="B", n=20):
    d = date(2025, 8, 1)
    ms = []
    for k in range(n):
        ms.append(Match("E0", "2526", d.replace(day=1 + k % 28), strong, weak, 3, 0, odds=(1.5, 4, 6)))
        ms.append(Match("E0", "2526", d.replace(day=1 + k % 28), weak, strong, 0, 2, odds=(5, 4, 1.6)))
    return ms


def test_fit_ranks_the_stronger_team_higher_and_uses_only_the_past():
    f = fit(_synthetic(), date(2025, 9, 1))
    i, j = f.teams.index("A"), f.teams.index("B")
    assert f.attack[i] > f.attack[j] and f.defence[i] < f.defence[j]
    lh, la = f.rates("A", "B")
    assert lh > la
    with pytest.raises(ValueError):
        fit(_synthetic(), date(2025, 7, 1))  # nothing before then


def test_prior_shrinks_small_samples_towards_average():
    f = fit(_synthetic(n=1), date(2025, 9, 1))
    assert 0.5 < f.attack[f.teams.index("A")] < 2  # one 3-0 win isn't enough to make A a juggernaut


def test_score_matches_hand_computation():
    rows = [((0.5, 0.3, 0.2), "H"), ((0.2, 0.3, 0.5), "D")]
    s = score(rows)
    assert s["called"] == 0.5
    assert s["log_loss"] == pytest.approx((-np.log(0.5) - np.log(0.3)) / 2, abs=1e-4)
    assert s["brier"] == pytest.approx(((0.25 + 0.09 + 0.04) + (0.04 + 0.49 + 0.25)) / 2, abs=1e-4)
