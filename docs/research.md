# Research notes

The question in every test here is how well a rating predicts games it has not seen.

## The test

Each model is fit on the games through week g and asked about the games of week g+1, for g from 2 on. The scored games are the 900 between two FBS teams from week 3 of 2025 through week 6 of 2026. A model's predicted margin is turned into a win probability by fitting one slope and a home edge, and the scores below come from ten-fold cross-validation grouped by week, so no game is scored by a fit that saw it. A coin flip has a log-loss of 0.693.

| Model | Log-loss | Winner picked |
|---|---|---|
| Margin rating with its preseason prior (the model before this change) | 0.566 | 69.9% |
| Points rating, offense and defense solved separately | 0.528 | 72.8% |
| Blend of 80% points rating and 20% margin rating (the model now) | 0.528 | 72.7% |
| Betting line, for reference | 0.516 | 74.2% |
| Betting line and the blend together | 0.513 | 73.9% |

The code is in `research/rule_eval.py`. `research/fetch_components.py` pulls the extra tables once, and the free CollegeFootballData key allows 1,000 calls a month, which that script uses a few dozen of.

## What changed in the model

The old rating compresses each margin with a logarithm and solves one number per team. The new points rating solves two numbers per team from the points each offense scored against each defense: the points scored equal an offense term minus a defense term plus a small site effect. Scores are capped at 56 so a single blowout does not run a rating. The solve leans on last season's ratings at the start of the year, with a share of 0.8 carried over and a ridge that weakens as games are played, and older games fade over 16 weeks instead of 8.

The house rule still holds. A top-group team's win over an outside team is ignored when the top-group ratings are re-solved, and a loss to an outside team counts in full. Adding the same amount to every offense and every defense changes no prediction, and with the outside wins removed only the prior fixed that amount, so the group's average level drifted away from the outside teams. Holding the group average at its first-stage value removed the drift. Before that fix the rule cost about 0.03 of log-loss, and after it the cost is under 0.001.

Strength is now the blend of the two ratings, in the margin rating's units: 20% margin rating plus 80% of 0.375 times net points per game. The weight comes from a grid over 0 to 1, where 0.8 and 0.9 tie. The blended rating feeds the resume, the schedule strength and the win probabilities, with a game noise of 5.75 and a home edge of the margin rating's edge times 0.2 minus 0.25.

## What did not help

Each row adds one more piece to the margin rating and the separate offense and defense ratings, scored the same way on games from week 4 on.

| Added | Log-loss |
|---|---|
| Nothing | 0.548 |
| Play value summed over a game (PPA), offense and defense | 0.548 |
| Play value and special teams value | 0.549 |
| Play value, special teams and the roster talent composite | 0.551 |

As the base measure for offense and defense, points scored beat the alternatives: 0.552 for points, 0.573 for summed play value, 0.573 for play value per play, 0.575 for success rate and 0.691 for explosiveness. Play value is still useful for one thing. The gap between the points rating and the play value rating shows when a scoreboard is running ahead of how the offense and defense actually played, and the tables carry it as `luck`.

Special teams and play value ratings are therefore reported beside each team and left out of the blend.

## Special teams

The measure is built from drives. A kickoff or punt is worth the expected points of the receiving team's starting yard line against what a kick of that type usually gives, a field goal is worth three points times the made kick minus the make probability for its distance, and returned touchdowns count in full. The drive feed mirrors its end-spot columns for many 2026 drives, so end spots come from start yardage minus yards gained. The ratings are shrunk hard, since a team's special teams play is a few events a game.

## Schedule strength

Strength of schedule is two measures now. The past schedule is the average rating of the opponents already played, with road opponents counted as stronger, and it feeds the score. The future schedule is the same average over the games still to come and is shown beside it.

## Where the rest of the gap sits

The betting line is still ahead by 0.012 of log-loss, and adding the line to the blend only moves it from 0.528 to 0.513. A line carries injuries, quarterback changes and weather that this model does not see, and on the site the line is printed next to the model's probability for each upcoming game.

## Rerunning it

```bash
python research/fetch_components.py
python research/rule_eval.py
python research/backtest.py
python research/stack.py
```

## What a loss costs in the board score

The board score is the resume (wins above a bubble team), the forward strength and the strength of the schedule played, as z-scores. It is a ranking of who has earned a place, so a loss should cost something. In week 6 of 2026, Alabama lost 17-20 at home to the 6-0 team that was rated second, and kept the #1 spot, still above the team that beat it. That looked wrong, so I checked how losses move the score across 2025 and 2026 (`research/rank_review.py`, `research/rank_review2.py`).

What the review found:

- Across the 91 losses by teams ranked in the top 25 the week before, the average team fell 7.5 places and 9% did not fall at all. Alabama's week 6 was among the cheapest of the 91, so the cost is not one team's quirk.
- The resume credits a result against what a bubble team would have done. Losing to a strong team is a game the bubble team would usually lose too, so the loss costs a fraction of a point (0.38 of the resume in the synthetic test, against a twin that lost nothing), where a win over a weak team is worth about as much.
- The schedule term pays again for the same fact. Playing a strong opponent already lifts the resume and then lifts the schedule score, win or lose. With the schedule term at 15% the average drop after a loss is 0.28 z, against 0.37 with no schedule term.
- Losses to stronger opponents cost 0.24 z and losses to weaker ones 0.35 z, which is the intended direction but too steep a discount.

How other systems handle it: ESPN's Strength of Record asks how likely an average top-25 team is to match a team's record against its schedule, so it counts wins and losses and not margins. Résumé SP+ projects the margin an average top-five team would have against the schedule and then takes 7 points off for every loss. Colley's method uses results only. FPI is a predictive rating and is not meant to rank who has earned a place. A common view of the résumé kind is that a win should never hurt a team and a loss should never help it. The score here let a loss to a strong team come out nearly free, which the flat penalty in Résumé SP+ does not.

The fix is a flat penalty per loss inside the resume, whoever the loss was to. Here is how it scores, with the schedule term at 15% (explained is the share of played games where the winner has the higher score; the log-loss is on next week's games between top-group teams, and a coin flip is 0.693):

| Penalty per loss | Explained, all games | Explained, top-25 vs top-25 | Log-loss | Alabama, week 7 |
|---|---|---|---|---|
| 0 (before) | 0.845 | 0.774 | 0.578 | 3rd |
| 1 | 0.842 | 0.773 | 0.578 | 7th |
| 2 (now) | 0.842 | 0.791 | 0.579 | 9th |
| 3 | 0.843 | 0.790 | 0.579 | 10th |

The penalty does not change how well the score predicts or explains games overall. Games are not scored by the board score anyway, the strength rating does that (0.544 on the same games, against 0.578 for the full score), and the board score is meant to be a résumé. It does improve the games between two top-25 teams, where it matters most, and it puts Georgia above Alabama in week 6. The penalty is 2 resume points, about one standard deviation of the resume. A schedule term of 8% with no penalty was close on the head-to-head measure but kept losses cheap.

This is a design judgment more than a fitted result: the data show the penalty is harmless to accuracy and helps the head-to-head cases, and the size comes from matching how voters and the other résumé systems treat a loss.

