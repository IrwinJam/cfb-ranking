"""Made-up seasons with known team strengths, used to test that the ratings recover them."""
import numpy as np
import pandas as pd

from .config import POWER_FOUR


class SyntheticProvider:
    def __init__(self, seed=3, weeks=12, per_conf=16, k=4.0):
        rng = np.random.default_rng(seed)
        self.p4 = [f"{c.replace(' ', '')}-{i:02d}" for c in POWER_FOUR for i in range(per_conf)]
        g5 = [f"G5-{i:02d}" for i in range(24)]
        teams = self.p4 + g5 + [f"FCS-{i:02d}" for i in range(10)]
        self.true = pd.Series(rng.normal(0, 8, len(teams)), teams)
        self._teams = pd.DataFrame(
            [(t, c) for c in POWER_FOUR for t in self.p4 if t.startswith(c.replace(" ", ""))] + [(t, "G5") for t in g5],
            columns=["school", "conference"])
        rows = []
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for a, b in zip(order[::2], order[1::2]):
                neutral = rng.random() < 0.08
                c = self.true[a] - self.true[b] + (0 if neutral else 3.0) + rng.normal(0, 8)
                m = np.sign(c) * (np.exp(abs(c) / k) - 1)
                rows.append((w, neutral, a, round(30 + m / 2), b, round(30 - m / 2)))
        self._games = pd.DataFrame(rows, columns=["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"])

    def games(self, year):
        return self._games

    def teams(self, year):
        return self._teams
