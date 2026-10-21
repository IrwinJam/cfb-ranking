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

## What a loss costs in the board score (a flat loss penalty was tried and reverted)

The board score is the resume (wins above a bubble team), the forward strength and the strength of the schedule played, as z-scores. In week 6 of 2026 Alabama lost 17-20 at home to the 6-0 team rated second and kept #1, still above the team that beat them. The scripts `research/rank_review.py`, `rank_review2.py` and `rank_review3.py` look at why, over 2025 and 2026.

What they found:

- Across the 91 losses by teams ranked in the top 25 the week before, the average team fell 7.5 places and 9% did not fall at all. Alabama's week 6 was among the mildest, so it is not one team.
- The resume scores a result against what a bubble team would have done. A bubble team usually loses to a strong team too, so that loss costs a fraction of a point.
- The schedule term then pays for the same strong opponent a second time, whether the game was won or lost. With the term the average drop after a loss is 0.28 z, without it 0.37.

A flat penalty of 2 resume points per loss (the idea behind Résumé SP+'s 7 points a loss) was tried. It kept the games explained and the log-loss where they were (0.578 against 0.579) and fixed the Alabama case, but it lifted Texas Tech, 6-0 against a weak schedule, to second, which no one believes. It was reverted: the parameter stays in the code at 0.

How closely each version matches the AP poll released after the same games (rank correlation among the teams the poll ranks, fitted on nothing, 2025 / 2026):

| Score | 2025 | 2026 |
|---|---|---|
| Current board score | 0.775 | 0.797 |
| Current plus a loss penalty of 2 | 0.804 | 0.813 |
| Without the separate schedule term | 0.813 | 0.833 |
| Resume 60% and strength 40% | 0.823 | 0.838 |
| Strength alone | 0.698 | 0.725 |
| Resume alone | 0.785 | 0.649 |

Taking the separate schedule term out agrees with the poll better in both seasons. It is also the logical part to take out: the resume already measures how hard the schedule was (it is wins against what a bubble team would win on that schedule), and the strength rating is already opponent-adjusted. The board now ranks with resume and strength only (schedule weight 0), mixed as `beta` of strength, which rises from 30% in week 1 to 50% from week 12. The Rankings Board has a Resume & strength page where the mix can be moved from resume only to strength only.
