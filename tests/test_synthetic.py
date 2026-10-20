"""Check on made-up data: the fitted ratings should track the true strengths. Run: python tests/test_synthetic.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aapr.config import Params
from aapr.pipeline import run_week
from aapr.synthetic import SyntheticProvider

prov = SyntheticProvider()
r = run_week(prov, 2025, 12, Params(), prior="none")["rankings"].set_index("team")
truth = prov.true.reindex(r.index)
corr = np.corrcoef(r.theta_margin, truth)[0, 1]
print(f"correlation of the margin ratings with true strengths after 12 weeks: {corr:.3f}")
assert corr > 0.85
blend = np.corrcoef(r.theta, truth)[0, 1]
print(f"correlation of the blended strength with true strengths: {blend:.3f}")
assert blend > 0.6

# a loss must cost something even against a strong team: two teams with identical results except one extra loss
import pandas as pd
from aapr.ratings import resume_scores
theta = pd.Series({"A": 5.0, "B": 5.0, "S": 8.0, "W": 0.0})
games = pd.DataFrame([dict(week=1, neutralSite=True, homeTeam="A", homePoints=20, awayTeam="W", awayPoints=10),
                      dict(week=1, neutralSite=True, homeTeam="B", homePoints=20, awayTeam="W", awayPoints=10),
                      dict(week=2, neutralSite=True, homeTeam="S", homePoints=30, awayTeam="A", awayPoints=20)])
rs = resume_scores(games, theta, 0.0, 5.0, 10.0, loss_penalty=0.0)
rp = resume_scores(games, theta, 0.0, 5.0, 10.0, loss_penalty=2.0)
print(f"resume of the team that lost to a strong team, without and with the penalty, against its twin: {rs['A'] - rs['B']:.2f}, {rp['A'] - rp['B']:.2f}")
assert rs["A"] - rs["B"] > -1.0                                              # the plain version costs under a point
assert (rp["A"] - rp["B"]) < (rs["A"] - rs["B"]) - 1.9                          # the penalty adds the full amount
