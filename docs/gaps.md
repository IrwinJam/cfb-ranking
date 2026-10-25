# Gaps

What the model does not do, and where its numbers are weaker than they look.

- **Injuries.** CollegeFootballData has no injury feed, so the model treats every player as available. The betting line is better than the blended rating by 0.012 log-loss, and injuries and quarterback news are the likely reason.
- **Sample size.** Eight weeks of games is thin. The game-results fit beats the defaults by 0.010 log-loss on held-out weeks.
- **Weakly fitted settings.** The score barely changes across the allowed range of the top-group game weight and the upset bonus, so their fitted values are simply where the search stopped. The strength of the preseason prior reached the edge of its allowed range.
- **Hand-set choices.** The 0.15 weight on schedule strength and the 0.5 blend between rating and resume are chosen by hand. Game results cannot identify either one.
- **Preseason prior.** It is last season's ratings, shrunk. Transfers and roster turnover are not modeled.
- **Outside teams.** Ratings for teams outside the top group rest on fewer games against top-group opponents.
- **Win counts.** The resume measure rewards the number of wins over top-group teams more than the quality of those teams, because a bubble team's chance in any one game sits near 50 to 65 percent. A scenario where a team wins out can therefore lift it further than a committee would.
- **Special teams.** The rating is built from kickoffs, punts and field goals in the drive feed, which has no kick distances or returner names. It did not improve any prediction, so it is shown and not used.
- **Points rating.** Scores are capped at 56 and the prior share of 0.8 was picked from two seasons of games. A team that runs up scores against weak defenses is helped by the offense term, and the cap is the only guard.
- **Not in this repo.** The playoff simulation and the what-if solver run on the author's website. The model here does not depend on either.
- **Not a prediction.** The ranking describes results so far. It does not forecast the final order.
