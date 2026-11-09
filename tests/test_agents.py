"""The three-agent preview: coordination over MCP, the fact-check loop, and replay without leakage."""
import asyncio
import re

import pytest

from matchday import agents
from matchday.agents import MAX_REVISIONS, NUMBER, check, figures, ordinal, run, streak, template_writer
from matchday.server import mcp

REPLAY = ("E0", "2026-11-07")


@pytest.fixture(scope="module")
def replay():
    return asyncio.run(run(*REPLAY, server=mcp))


def ledger_ok(r: dict) -> list[str]:
    """Re-check the published preview from outside: every number per section must be in that fixture's ledger."""
    bad = []
    head, *sections = r["preview"].split("\n### ")
    for sec in sections:
        title, body = sec.split("\n\n", 1)
        a = next(a for a in r["angles"] if title.startswith(a["key"]))
        allowed = {*a["facts"].values(), str(int(a["date"][8:])), "2.5"}
        bad += [f"{a['key']}: {t}" for t in NUMBER.findall(body) if t not in allowed]
    return bad


def test_replay_covers_every_fixture_with_no_tool_errors(replay):
    assert replay["fixtures"] == 9 == replay["preview"].count("\n### ")
    assert replay["tool_errors"] == [] and replay["tool_calls"] >= 9 * 5
    assert replay["revisions"] == 0 and replay["cut"] == []


def test_every_published_number_is_in_the_ledger(replay):
    assert ledger_ok(replay) == []


def test_replay_never_asks_for_data_after_kickoff():
    calls = []
    orig = agents.Tools.__call__

    async def spy(self, name, **args):
        calls.append((name, args))
        return await orig(self, name, **args)

    agents.Tools.__call__ = spy
    try:
        asyncio.run(run(*REPLAY, server=mcp))
    finally:
        agents.Tools.__call__ = orig
    for name, args in calls:
        if name in {"predict_match", "team_form", "head_to_head"}:
            assert args.get("asof") and args["asof"] >= REPLAY[1], (name, args)
        if name == "standings" and "asof" in args:
            assert args["asof"] < REPLAY[1]


def test_check_flags_invented_figures_and_wrong_teams():
    allowed = {"62%", "1-0", "1st", "24"}
    teams, league = {"Arsenal", "Hull"}, {"Arsenal", "Hull", "Chelsea", "Man City"}
    assert check("Arsenal (1st, 24 pts) are 62% favourites; 1-0 the likeliest score.", allowed, teams, league) == []
    assert check("Arsenal are 71% favourites.", allowed, teams, league) == ["unsupported figure '71%'"]
    assert check("Arsenal won 3-0 last time.", allowed, teams, league) == ["unsupported figure '3-0'"]
    assert check("Hull fans will remember Chelsea.", allowed, teams, league) == ["names Chelsea, who isn't in this fixture"]


def test_editor_sends_a_hallucinating_draft_back_and_accepts_the_fix():
    seen = []

    def writer(a, issues):
        seen.append(list(issues))
        text = template_writer(a, issues)
        return text if issues else text + " They have won 97% of games since 1888."

    r = asyncio.run(run(*REPLAY, server=mcp, writer=writer))
    assert r["revisions"] == 1 and r["cut"] == []
    assert any("unsupported figure '97%'" in i for batch in seen for i in batch)
    assert "97%" not in r["preview"] and ledger_ok(r) == []


def test_a_writer_that_never_fixes_it_gets_its_sentence_cut():
    def writer(a, issues):
        return template_writer(a, issues) + " They have won 97% of games since 1888."

    r = asyncio.run(run(*REPLAY, server=mcp, writer=writer))
    assert r["revisions"] == MAX_REVISIONS
    assert len(r["cut"]) == r["fixtures"] and all("97%" in s for s in r["cut"])
    assert "97%" not in r["preview"] and ledger_ok(r) == []


@pytest.mark.parametrize("form,want", [("LWWWW", ("W", 4)), ("WWLLL", ("L", 3)), ("D", ("D", 1)), ("", ("", 0))])
def test_streak(form, want):
    assert streak(form) == want


@pytest.mark.parametrize("n,s", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (21, "21st")])
def test_ordinal(n, s):
    assert ordinal(n) == s


def test_digits_in_team_names_are_not_figures():
    # Schalke 04 once had every sentence it appeared in flagged, rewritten twice and cut
    assert figures("Schalke 04 (17th, 6 pts) host Mainz.", {"Schalke 04", "Mainz"}) == ["17th", "6"]
    assert check("Schalke 04 have lost 3 in a row.", {"3"}, {"Schalke 04", "Mainz"}, {"Schalke 04", "Mainz", "Bayern Munich"}) == []


def test_numbers_regex_reads_scores_percentages_and_ordinals():
    assert NUMBER.findall("1st on 24 pts, 62% to win, 1-0, xG 1.65, over 2.5") == ["1st", "24", "62%", "1-0", "1.65", "2.5"]
    assert re.sub(NUMBER, "#", "Arsenal 3-1") == "Arsenal #"
