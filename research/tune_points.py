"""Tune the points-based offense/defense rating: recency, ridge strength, a cap on lopsided scores."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr import components as C
from aapr.cfbd_client import CFBD

api = CFBD(ttl_hours=24 * 365)
YEARS = {2025: range(1, 17), 2026: range(1, 8)}
raw = {y: C.load_team_games(api, y, w) for y, w in YEARS.items()}
games = {y: C.load_games(api, y) for y in YEARS}


def run(tau, kappa0, cap, col="pts"):
    rows = []
    for y in YEARS:
        TG, G = raw[y].copy(), games[y]
        if cap:
            TG["pts_c"] = TG.pts.clip(upper=cap)
        col_ = "pts_c" if cap else col
        for wk in range(4, int(G.week.max())):
            nxt = G[G.week == wk + 1]
            if nxt.empty:
                continue
            O, D, mu, b = C.fit_off_def(TG[TG.week <= wk], col_, wk, tau=tau, kappa0=kappa0)
            nr = O + D
            for r in nxt.itertuples():
                if r.homeTeam in nr.index and r.awayTeam in nr.index and r.homePoints != r.awayPoints:
                    rows.append((y * 100 + wk + 1, nr[r.homeTeam] - nr[r.awayTeam], 0.0 if r.neutralSite else 1.0, float(r.homePoints > r.awayPoints)))
    d = pd.DataFrame(rows, columns=["g", "od", "site", "win"])
    X, y, g = d[["od", "site"]].to_numpy(), d.win.to_numpy(), d.g.to_numpy()
    ll = 0.0
    for tr, te in GroupKFold(n_splits=10).split(X, y, g):
        p = np.clip(LogisticRegression(C=100, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum()
    return ll / len(d), len(d)


print("tau   kappa0  cap    log-loss  games")
best = None
for tau, kappa0, cap in itertools.product((12, 16, 24), (0.5, 1, 2, 3), (56, 50, 63)):
    ll, n = run(tau, kappa0, cap)
    print(f"{tau:<5g} {kappa0:<7g} {str(cap):6s} {ll:.4f}  {n}", flush=True)
    if best is None or ll < best[0]:
        best = (ll, tau, kappa0, cap)
print("best", best)
