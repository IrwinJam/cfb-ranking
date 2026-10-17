"""Choose the settings of the points-based offense/defense rating and how much of it to blend into the margin rating.

Both seasons, each with the previous season as its preseason prior (2024 for 2025, 2025 for 2026).
"""
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr import components as C
from aapr.cfbd_client import CFBD, CFBDProvider
from aapr.config import INDEPENDENT_TEAMS, TOP_GROUP
from aapr.params_io import load_params
from aapr.pipeline import preseason_prior
from aapr.ratings import fit_ratings

api = CFBD(ttl_hours=24 * 365)
prov = CFBDProvider(ttl_hours=24 * 365)
P = load_params("fitted", quiet=True)
G = {y: C.load_games(api, y) for y in (2024, 2025, 2026)}
TG = {y: C.team_games_from_games(G[y]) for y in G}
TAU = 16.0

grid_share, grid_k = (0.4, 0.6, 0.8, 1.0), (1.0, 3.0, 6.0, 10.0)
rows = []
for y in (2025, 2026):
    g = G[y]
    conf = {}
    for r in g.itertuples():
        conf[r.homeTeam], conf[r.awayTeam] = r.homeConference, r.awayConference
    p4 = {t for t, c in conf.items() if c in TOP_GROUP} | set(INDEPENDENT_TEAMS)
    theta0 = preseason_prior(prov, y, P, "season", p4)
    O0, D0, _, _ = C.fit_off_def(TG[y - 1], "pts_c", int(G[y - 1].week.max()), tau=1e9, kappa0=0.4)
    for wk in range(2, int(g.week.max())):
        nxt = g[g.week == wk + 1]
        if nxt.empty:
            continue
        past = g[g.week <= wk]
        th, h, _ = fit_ratings(past[["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]], wk, P, theta0, p4)
        fits = {}
        for sh, k0 in itertools.product(grid_share, grid_k):
            O, D, mu, b = C.fit_off_def(TG[y][TG[y].week <= wk], "pts_c", wk, tau=TAU, kappa0=k0, prior=(sh * O0, sh * D0))
            fits[(sh, k0)] = O + D
        for r in nxt.itertuples():
            if r.homePoints == r.awayPoints:
                continue
            site = 0.0 if r.neutralSite else 1.0
            row = dict(gameId=r.gameId, year=y, fbs2=float(r.homeClassification == 'fbs' and r.awayClassification == 'fbs'), g=y * 100 + wk + 1, win=float(r.homePoints > r.awayPoints), site=site,
                       theta=th.get(r.homeTeam, np.nan) - th.get(r.awayTeam, np.nan) + h * site)
            for key, nr in fits.items():
                row[f"od_{key[0]}_{key[1]}"] = nr.get(r.homeTeam, np.nan) - nr.get(r.awayTeam, np.nan)
            rows.append(row)
d = pd.DataFrame(rows).dropna().reset_index(drop=True)
d = d[d.fbs2 == 1.0].reset_index(drop=True)      # only games between two FBS teams
print(len(d), "games tested,", (d.year == 2025).sum(), "in 2025 and", (d.year == 2026).sum(), "in 2026")
d.to_pickle(Path(__file__).with_name("final_eval_games.pkl"))


def cv(cols, data=None):
    dd = d if data is None else data
    X, y, g = dd[cols].to_numpy(float), dd.win.to_numpy(), dd.g.to_numpy()
    ll = acc = 0.0
    for tr, te in GroupKFold(n_splits=10).split(X, y, g):
        pr = np.clip(LogisticRegression(C=100, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr)).sum()
        acc += ((pr > 0.5) == (y[te] > 0.5)).sum()
    return ll / len(dd), acc / len(dd)


print("\nmargin ridge with its preseason prior (current model):  log-loss %.4f  accuracy %.3f" % cv(["theta", "site"]))
print("\npoints rating by prior share and ridge strength (log-loss):")
print("share \ ridge " + "".join(f"{k:>9g}" for k in grid_k))
best = (9, None)
for sh in grid_share:
    line = f"{sh:<13g}"
    for k0 in grid_k:
        ll, _ = cv([f"od_{sh}_{k0}", "site"])
        line += f"{ll:9.4f}"
        if ll < best[0]:
            best = (ll, (sh, k0))
    print(line)
print("best", best)
sh, k0 = best[1]
col = f"od_{sh}_{k0}"
s = float(LinearRegression().fit(d[[col]], d.theta).coef_[0])
d["od_t"] = s * d[col]
print(f"\ntheta units per point of net rating: {s:.3f}")
for w in (0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 1.0):
    d["blend"] = (1 - w) * d.theta + w * d.od_t
    ll, acc = cv(["blend", "site"])
    print(f"weight on the points rating {w:.2f}:  log-loss {ll:.4f}  accuracy {acc:.3f}")


# against the betting line, on the games that have one
rowsl = []
for y in (2025, 2026):
    for r in api.get("/lines", year=y, seasonType="regular"):
        sp = [x["spread"] for x in r["lines"] if x.get("spread") is not None]
        if sp:
            rowsl.append((r["id"], -float(np.mean(sp))))
m = d.merge(pd.DataFrame(rowsl, columns=["gameId", "market"]), on="gameId", how="inner")
w = 0.8
m["model"] = (1 - w) * m.theta + w * m.od_t
print(f"{len(m)} games with a betting line (FBS vs FBS, weeks 3 and later)")
for name, cols in (("margin ridge (old)", ["theta", "site"]), ("blend (new)", ["model", "site"]), ("betting line", ["market"]), ("line + new blend", ["market", "model"])):
    ll, acc = cv(cols, m)
    print(f"  {name:22s} log-loss {ll:.4f}  accuracy {acc:.3f}")
