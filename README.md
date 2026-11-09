# matchday-mcp

An MCP server that gives any AI agent the top five European leagues as tools: results, tables, form,
head-to-heads, bookmaker odds, and a Poisson match model with a walk-forward track record.

![ci](https://github.com/otawfik/matchday-mcp/actions/workflows/ci.yml/badge.svg)

**Stack:** Python 3.12 · MCP Python SDK 2.x (`MCPServer`, stdio and streamable HTTP) · Pydantic
(typed tool output) · NumPy · pytest · GitHub Actions

Ask Claude *"Who's in form in the Premier League, and what does your model give Arsenal v Man City?"*
and it calls `standings`, `team_form` and `predict_match`, then reads typed JSON back:

```json
{"home": "Barcelona", "away": "Real Madrid", "p_home": 0.532, "p_draw": 0.206, "p_away": 0.263,
 "expected_goals": [2.18, 1.5], "likely_scores": [{"score": "2-1", "p": 0.09}, {"score": "1-1", "p": 0.083}],
 "p_over_2_5": 0.71, "p_both_score": 0.688, "fitted_on": 499}
```

## Tools

| tool | what it returns |
|---|---|
| `list_leagues` | leagues and seasons loaded, match counts, latest result |
| `standings` | the table (points, goal difference, goals), optionally rewound to any date |
| `team_form` | last N results from the team's side, with xG where available |
| `head_to_head` | every meeting in the loaded seasons, and the record |
| `predict_match` | win/draw/loss, expected goals, likeliest scores, over 2.5, both teams to score |
| `market_odds` | bookmaker average odds, as probabilities with the margin removed |
| `upcoming_fixtures` | scheduled fixtures with odds |
| `model_track_record` | the backtest below, so an agent can say how far to trust a forecast |

Names are matched loosely ("Manchester United" → Man United, "spurs" → Tottenham, "arsnal" → Arsenal).
A name that can't be resolved comes back as a tool error that lists the closest teams, so the model
can correct itself instead of guessing.

## How good is the model?

Walk-forward: before each match day, the model is refitted on every earlier match and predicts that
day. It never sees a result before forecasting it. 2,007 matches across five leagues, 2025-26 and
2026-27 (`reports/backtest.json`, recomputed and checked in CI):

| | log loss ↓ | Brier ↓ | results called |
|---|---|---|---|
| uniform (⅓ each) | 1.099 | 0.667 | – |
| league's home/draw/away rates | 1.078 | 0.653 | 43.8% |
| always pick the home side | – | – | 43.9% |
| **Poisson model** | **1.007** | **0.602** | **50.7%** |
| bookmakers (average odds, margin removed) | 0.975 | 0.580 | 53.9% |

The model beats every naive baseline, and CI fails if it ever stops doing so. It doesn't beat the
bookmakers, and it isn't expected to: their prices carry team news, injuries and money this model
never sees. The gap (0.032 in log loss) is the honest size of what goals alone leave out.

**Does xG help?** Only this season's files carry xG, so this is measured on 2026-27's 486 matches:

| fitted on | log loss | Brier | called |
|---|---|---|---|
| goals | 0.977 | 0.581 | 52.9% |
| xG | 0.981 | 0.583 | 53.1% |
| half goals, half xG | **0.976** | **0.579** | **53.9%** |
| bookmakers | 0.944 | 0.558 | 56.6% |

A small gain from blending. It's within noise at this sample size, so the server ships the goals fit.

**The model:** Maher-style Poisson. Each team has an attack and a defence strength, plus a league
home factor, fitted by weighted maximum likelihood. Matches decay with a 120-day half-life, so this
season outweighs last. A four-match pseudo-count prior pulls teams with little data (promoted sides in
August) towards league average. `predict_match(asof=...)` fits on matches before that date only,
which is what makes the backtest honest.

## Use it

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -e .

pytest -q                           # 34 tests, including a real stdio MCP session
python -m matchday.backtest --check # walk-forward backtest; exits 1 if the model stops beating the baselines
python -m matchday.refresh          # pull the latest results and fixtures (the snapshot is committed)
matchday-mcp                        # stdio
matchday-mcp --http 8765            # streamable HTTP
```

**Claude Code:** `claude mcp add matchday -- /path/to/matchday-mcp/.venv/bin/matchday-mcp`

**Claude Desktop** (`claude_desktop_config.json`):

```json
{"mcpServers": {"matchday": {"command": "/path/to/matchday-mcp/.venv/bin/matchday-mcp"}}}
```

## Tests

- **Data:** every league and season loads, odds are valid, and team-name and league-name resolution
  work, including the "did you mean" path.
- **Model:** the score grid is a probability distribution, and the margin removal sums to 1. On
  synthetic data the fit ranks the stronger team higher and refuses to fit with no past matches. The
  prior stops one 3-0 from making a team a juggernaut. The scorer matches a hand calculation.
- **Server:** every tool is called through an MCP client session. The checks cover the tool schemas,
  that a table adds up (goals for = goals against, points = 3W + D), that `asof` rewinds, that
  predictions are walk-forward, and that bad input becomes a readable tool error. One test spawns
  `python -m matchday.server` as a subprocess and speaks JSON-RPC over stdio, the way Claude Desktop
  does.

## Next

A multi-agent layer on top: a scout, an analyst and an editor that coordinate through these tools to
write a matchday preview, triggered by a webhook when a fixtures file updates.

Data: [football-data.co.uk](https://www.football-data.co.uk) (results, match statistics and odds), free
for non-commercial use.
