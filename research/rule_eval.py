"""What does the house rule cost? Points rating with and without ignoring a top-group team's wins over outside teams."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
rows = []
for y in (2025, 2026):
    g = G[y]
    conf = {}
    for r in g.itertuples():
        conf[r.homeTeam], conf[r.awayTeam] = r.homeConference, r.awayConference
    p4 = {t for t, c in conf.items() if c in TOP_GROUP} | set(INDEPENDENT_TEAMS)
    theta0 = preseason_prior(prov, y, P, "season", p4)
    last_prev = int(G[y - 1].week.max())
    Oj, Dj, _, _ = C.fit_points(TG[y - 1], last_prev, tau=1e9, kappa0=0.4)
    Og, Dg, _, _ = C.fit_points(TG[y - 1], last_prev, tau=1e9, kappa0=0.4, group={t for t in p4})
    for wk in range(2, int(g.week.max())):
        nxt = g[g.week == wk + 1]
        if nxt.empty:
            continue
        past = g[g.week <= wk]
        th, h, _ = fit_ratings(past[["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]], wk, P, theta0, p4)
        tgw = TG[y][TG[y].week <= wk]
        res = {}
        for name, grp, pr in (("joint", None, (0.8 * Oj, 0.8 * Dj)), ("rule", p4, (0.8 * Og, 0.8 * Dg))):
            O, D, mu, b = C.fit_points(tgw, wk, tau=16.0, kappa0=6.0, prior=pr, group=grp)
            res[name] = O + D
        for r in nxt.itertuples():
            if r.homePoints == r.awayPoints or not (r.homeClassification == "fbs" and r.awayClassification == "fbs"):
                continue
            site = 0.0 if r.neutralSite else 1.0
            row = dict(gameId=r.gameId, g=y * 100 + wk + 1, win=float(r.homePoints > r.awayPoints), site=site,
                       theta=th.get(r.homeTeam, np.nan) - th.get(r.awayTeam, np.nan) + h * site,
                       homeG=float(r.homeTeam in p4), awayG=float(r.awayTeam in p4))
            for k, nr in res.items():
                row[k] = nr.get(r.homeTeam, np.nan) - nr.get(r.awayTeam, np.nan)
            rows.append(row)
d = pd.DataFrame(rows).dropna().reset_index(drop=True)
print(len(d), "FBS vs FBS games")


def cv(cols, dd=None):
    dd = d if dd is None else dd
    X, y, g = dd[cols].to_numpy(float), dd.win.to_numpy(), dd.g.to_numpy()
    ll = acc = 0.0
    for tr, te in GroupKFold(n_splits=10).split(X, y, g):
        pr = np.clip(LogisticRegression(C=100, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr)).sum()
        acc += ((pr > 0.5) == (y[te] > 0.5)).sum()
    return ll / len(dd), acc / len(dd)


for label, cols in (("margin ridge (old)", ["theta", "site"]), ("points, every game counts", ["joint", "site"]), ("points, house rule", ["rule", "site"])):
    print(f"{label:30s} log-loss %.4f  accuracy %.3f" % cv(cols))
for w in (0.6, 0.8, 0.9):
    d["bl"] = (1 - w) * d.theta + w * 0.375 * d.rule
    print(f"blend with house rule, weight {w}:   log-loss %.4f  accuracy %.3f" % cv(["bl", "site"]))
print("\ngames between two top-group teams only:")
both = d[(d.homeG == 1) & (d.awayG == 1)]
for label, cols in (("margin ridge (old)", ["theta", "site"]), ("points, every game counts", ["joint", "site"]), ("points, house rule", ["rule", "site"])):
    print(f"{label:30s} log-loss %.4f  accuracy %.3f   ({len(both)} games)" % cv(cols, both.reset_index(drop=True)))


# against the betting line, on the games that have one
rowsl = []
for y in (2025, 2026):
    for r in api.get("/lines", year=y, seasonType="regular"):
        sp = [x["spread"] for x in r["lines"] if x.get("spread") is not None]
        if sp:
            rowsl.append((r["id"], -float(np.mean(sp))))
m = d.merge(pd.DataFrame(rowsl, columns=["gameId", "market"]), on="gameId", how="inner")
m["bl"] = 0.2 * m.theta + 0.8 * 0.375 * m.rule
print(f"{len(m)} games with a betting line")
for label, cols in (("margin ridge (old)", ["theta", "site"]), ("blend (new)", ["bl", "site"]), ("betting line", ["market"]), ("line and blend", ["market", "bl"])):
    print(f"{label:30s} log-loss %.4f  accuracy %.3f" % cv(cols, m))
