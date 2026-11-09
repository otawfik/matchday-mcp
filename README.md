# matchday-mcp

An MCP server that gives any AI agent the top five European leagues as tools: results, tables, form,
head-to-heads, bookmaker odds, and a Poisson match model with a walk-forward track record. On top of it,
three agents (a scout, an analyst and an editor) coordinate through those tools to write a fact-checked
matchday preview whenever a signed webhook fires.

![ci](https://github.com/otawfik/matchday-mcp/actions/workflows/ci.yml/badge.svg)

**Stack:** Python 3.12 · MCP Python SDK 2.x (`MCPServer`, stdio and streamable HTTP) · LangGraph ·
FastAPI · Pydantic (typed tool output) · NumPy · pytest · GitHub Actions

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

## The agents

```
webhook ─► scout ─► analyst ─► editor: write ─► editor: check ─┬─► preview
             │                      ▲                          │
          MCP tools                 └── unsupported claims ────┘  at most 2 rewrites, then cut
```

A LangGraph graph (`matchday/agents.py`). The agents share state, and they reach the data only through
the MCP server, over stdio by default (the way Claude Desktop does) or streamable HTTP with
`MATCHDAY_MCP_URL`.

- **Scout** works out the matchday's fixtures and builds a dossier for each: the forecast, market odds,
  both teams' form, the table and the head-to-head. That's about five tool calls per fixture.
- **Analyst** finds the angles: where the model and the bookmakers disagree by more than 8 points,
  winning and losing runs, and top-four clashes. It ranks fixtures by interest and writes a **ledger**
  of every figure the copy is allowed to state, already formatted.
- **Editor** writes the preview, then fact-checks it. Every number must be in that fixture's ledger,
  and every team it names must be playing in the fixture. A failing draft goes back to the writer with
  the list of problems. If two rewrites don't fix it, the unsupported sentences are cut rather than
  published.

The writer is a deterministic template, so the check never has to fire on it. The check exists so a
model-written draft (any function with the `Writer` signature) can't publish a number nobody looked up.
The tests drive the loop with a writer that invents "97% of games since 1888": once it gets fixed on
the rewrite, and once it never gets fixed and the sentence gets cut.

**Replay.** `--start 2026-11-07` previews a past matchday as if it were upcoming. Every tool call carries
`asof` = kickoff day, which a test enforces, so the agents can't see the results they're previewing.
That makes the agents gradeable against what actually happened:

```bash
python -m matchday.agents --league E0 --start 2026-11-07
```

```
## Premier League preview: Sat 7 Nov to Sun 8 Nov

9 fixtures. Most confident call: Arsenal (62%). Biggest split with the bookmakers: Arsenal v Hull (20 points).

### Arsenal v Hull · Sat 7 Nov

Arsenal (1st, 24 pts) host Hull (11th, 11 pts). Arsenal have won 4 in a row. Form, oldest to newest:
Arsenal LWWWW, Hull LDLDD. The model makes Arsenal favourites at 62% (draw 24%, Hull 14%), and 1-0 is
the likeliest score. Expected goals 1.65 to 0.64; 40% for over 2.5. The bookmakers disagree: the model
rates Arsenal 20 points less likely than the market does (62% against 82%).
...
```

## Evaluating the agents

`python -m matchday.evaluate --check` replays every matchday of 2026-27 from late August in all five
leagues. That's 49 runs and 445 fixtures, run in CI on every push (`reports/agents.json`):

| | |
|---|---|
| fixtures previewed / played | **445 / 445** |
| published numbers traced to a tool output (re-checked from outside the agents) | **7,387 / 7,387** |
| planted errors caught: a team from another match named | **442 / 442** |
| planted errors caught: one figure changed | **94.6%** of 445 |
| tool calls per fixture | 5.3 |
| median run (one league's matchday, in-process server) | 0.05 s |

The mutation test is the honest one. In every published section it changes one figure, or names a
team from another match, and checks that the editor notices. The 5.4% of changed figures it misses
are collisions. The ledger is a set per fixture, so a draw at 24% nudged to 31% passes if 31% is
already a real number in that fixture, such as the away side's chance. Catching those needs the check
to know which figure each sentence is about, not just whether the figure exists.

The evaluation caught a real bug on its first run. **Schalke 04** has digits in its name, so the check
read "04" as an invented figure. Every Schalke sentence was rewritten twice and then cut from the
preview. Team names are now masked before figures are extracted, and the gate requires zero rewrites
of the template's own copy.

What the replays say about the calls themselves:

- **When the model is confident, it's underconfident.** 76 calls at 60% or more (67% on average) were
  won 80% of the time. The sample is small, but it points the same way as the Poisson model's known
  habit of flattening big favourites.
- **When the model argues with the bookmakers, side with the bookmakers.** On the 159 fixtures where
  they were 8+ points apart, the market's log loss was 0.918 against the model's 0.997, and the model
  was closer only 30% of the time. That's why the preview reports disagreements as a talking point,
  not a tip.
- Three promoted Bundesliga sides (Elversberg, Paderborn, Schalke 04) had no history in the data
  before their first game, so those previews ran without a forecast. The run report lists them
  separately from tool errors.

## Webhook

```bash
MATCHDAY_WEBHOOK_SECRET=... uvicorn matchday.webhook:app --port 8000
```

`POST /webhooks/matchday` with `{"league": "E0"}` (or `"start"` to replay) returns `202` and a run id
at once. The agents run in the background, and `GET /runs/{id}` returns the preview and the run report
(tool calls, rewrites, anything cut). Requests are signed like GitHub webhooks: an HMAC-SHA256 of the
raw body in `X-Matchday-Signature`. A bad signature is a `401` before the body is parsed. With no secret
configured, the endpoint refuses everything rather than running open. `X-Matchday-Delivery` makes
redelivery idempotent, so the same id returns the same run instead of starting another.

## Use it

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -e .

pytest -q                           # 58 tests, including a real stdio MCP session
python -m matchday.backtest --check # walk-forward backtest; exits 1 if the model stops beating the baselines
python -m matchday.evaluate --check # replay the season through the agents
python -m matchday.agents --league E0  # preview the next matchday
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
- **Agents:** a replay covers every fixture with no tool errors. Every published number is in its
  ledger. Replay never asks for data after kickoff. The rewrite loop and the cut path both work, and
  digits in team names aren't counted as figures.
- **Webhook:** it refuses without a secret, rejects a bad signature and an unknown league, and a
  redelivered event returns the original run. A signed event goes all the way to a preview.
- **Server:** every tool is called through an MCP client session. The checks cover the tool schemas,
  that a table adds up (goals for = goals against, points = 3W + D), that `asof` rewinds, that
  predictions are walk-forward, and that bad input becomes a readable tool error. One test spawns
  `python -m matchday.server` as a subprocess and speaks JSON-RPC over stdio, the way Claude Desktop
  does.

## Next

- A model-written preview behind the same fact-check: Claude through the `Writer` interface, so the
  mutation test and the cut path guard real model output.
- A GitHub Actions schedule that runs `matchday.refresh` and fires the webhook when new fixtures land.

Data: [football-data.co.uk](https://www.football-data.co.uk) (results, match statistics and odds), free
for non-commercial use.
