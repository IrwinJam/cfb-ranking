# cfb-rankings

A ratings model for college football. It ranks the four Power Four conferences and Notre Dame from game results alone. Every team's strength is solved at once with ridge regression, each team also gets offense, defense and special teams ratings, and the model's settings are chosen by how well they predict games it has not seen.

## The model

Each game is one equation. The compressed margin, `k * sign * ln(1 + |margin|)`, equals the home rating minus the away rating plus a home-field edge. All FBS and FCS teams on the schedule are solved together. Older games fade with `exp(-weeks ago / tau)`, and every rating is pulled toward last season's rating with a strength that weakens as games are played.

A second pass rates the 68 top-group teams with every other rating held fixed. A top-group win over an outside team gets zero weight and a loss to one counts in full. Games between two top-group teams get extra weight. A scaled home-field edge makes road wins worth more, and a win by the lower-rated team gets a small bonus.

A second rating comes from the points each offense scores against each defense, so offense and defense are solved separately, in points per game against an average FBS team. The two ratings are blended, 80 percent from the points rating, and that blend is the strength used everywhere below. On games between two FBS teams from 2025 and 2026 it picked 72.7 percent of winners against 69.9 percent for the margin rating alone. The tests are in [docs/research.md](docs/research.md).

Special teams is rated from drive data: kickoff and punt field position, returns, and field goals against the make rate for their distance. A play-value (PPA) rating of offense and defense sits beside the points rating, and the gap between the two is reported as `luck`. Neither improved the predictions, so both are shown and left out of the blend.

The ranking score blends the rating with a resume measure, which is wins above what a bubble team would have won, and adds a small term for the average rating of past opponents. Opponents are always valued at the current week's ratings, so a schedule is re-graded as the season goes on. Strength of schedule is kept as two numbers: the past schedule, which feeds the score, and the future schedule, which is shown beside it.

## Choosing the settings

The settings were fit twice. The first fit used game results: ratings through week g predict the games of week g+1, scored by log-loss over 194 games involving a top-group team. The second fit matched the AP poll. Each was then scored on weeks it had not been fit on: one week is held out, the settings are refit on the rest, and that week is predicted.

| Settings | Mean held-out log-loss |
|---|---|
| Fit to game results | 0.584 |
| Hand-picked defaults | 0.592 |
| Fit to the AP poll | 0.639 |

These settings belong to the margin rating, and they were fit before the points rating was added. A coin flip scores 0.693 and lower is better. The AP fit matches the poll more closely and predicts games worse, which is why the game-results fit is the one in use (`fitted_params.json`). The AP fit is kept for comparison in `fitted_params_ap.json`.

## Rankings through week 6, 2026

| # | Team | Record |
|---|------|--------|
| 1 | Alabama (SEC) | 5-1 |
| 2 | Georgia (SEC) | 6-0 |
| 3 | Notre Dame | 6-0 |
| 4 | BYU (Big 12) | 5-0 |
| 5 | Texas Tech (Big 12) | 5-0 |
| 6 | Indiana (Big Ten) | 6-0 |
| 7 | Miami (ACC) | 5-0 |
| 8 | Texas (SEC) | 4-1 |
| 9 | Kentucky (SEC) | 5-1 |
| 10 | UCLA (Big Ten) | 5-0 |

<!-- ap:start -->
## AP Top 25 and the board, through week 6

The AP poll released after week 6 games, with the board's rank for each team. A team outside the Power Four group has no board rank. A positive gap means the board has the team higher.

| AP | Team | Board | Gap |
|---|------|-------|-----|
| 1 | Georgia | 2 | -1 |
| 2 | Notre Dame | 3 | -1 |
| 3 | Miami | 7 | -4 |
| 4 | Ohio State | 15 | -11 |
| 5 | Indiana | 6 | -1 |
| 6 | Alabama | 1 | +5 |
| 7 | Texas | 8 | -1 |
| 8 | BYU | 4 | +4 |
| 9 | Ole Miss | 13 | -4 |
| 10 | Texas Tech | 5 | +5 |
| 11 | Utah | 11 | 0 |
| 12 | UCLA | 10 | +2 |
| 13 | Missouri | 19 | -6 |
| 14 | Tennessee | 24 | -10 |
| 15 | Kentucky | 9 | +6 |
| 16 | Mississippi State | 17 | -1 |
| 17 | Iowa | 14 | +3 |
| 18 | Oklahoma State | 26 | -8 |
| 19 | USC | 25 | -6 |
| 20 | Houston | 16 | +4 |
| 21 | Wake Forest | 12 | +9 |
| 22 | LSU | 22 | 0 |
| 23 | Boise State | - | - |
| 24 | SMU | 32 | -8 |
| 25 | Northwestern | 20 | +5 |
<!-- ap:end -->

The weekly tables, schedule strength included, and the history of the fitted settings are in `outputs/2026/`. What the model leaves out is in [docs/gaps.md](docs/gaps.md).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env              # add a free key from collegefootballdata.com/key
python season_history.py --year 2026
python fit_to_games.py --year 2026 --through 6
python research/rule_eval.py      # the walk-forward test behind the blend
python tests/test_synthetic.py    # made-up seasons with known strengths, no key needed
```

## Files

- `aapr/ratings.py` holds the regression, the resume measure and the win probability.
- `aapr/components.py` holds the points rating, the special teams measure and the blend.
- `aapr/pipeline.py` builds one week's ranking, and `aapr/schedule.py` builds the schedule tables.
- `fit_to_games.py` and `fit_to_ap.py` fit the settings, with a leave-one-week-out check.
- `season_history.py` runs every completed week and writes the tables.
- `research/` holds the backtests behind the blend, and `docs/research.md` explains them.

Game results, schedules and conferences come from [CollegeFootballData.com](https://collegefootballdata.com). The API key is read from a local `.env` and never written anywhere else.
