"""Second pass on the board score: what a loss costs, and what a per-loss penalty would do. Reads rank_review_tabs.pkl."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tabs = pd.read_pickle(Path(__file__).with_name("rank_review_tabs.pkl"))
from aapr.params_io import load_params
P = load_params("fitted", quiet=True)


def z(s):
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def losses_through(t):
    g = t["g"]; g = g[g.week <= t["week"]]
    L = pd.concat([g.assign(t=g.homeTeam, l=g.homePoints < g.awayPoints), g.assign(t=g.awayTeam, l=g.awayPoints < g.homePoints)]).groupby("t")["l"].sum()
    return L.reindex(t["teams"]).fillna(0.0)


def score(t, gamma, c, rho=0.0):
    wk = t["week"]
    b = P.beta_early + (P.beta_late - P.beta_early) * min(wk / P.beta_ramp_weeks, 1)
    res = t["resume"] - c * losses_through(t)
    base = (1 - b) * z(res) + b * z(t["R"])
    return (1 - gamma) * base + gamma * z(t["sos"])


def run(gamma, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rowsP, cases = [], {}
    prev, drops, wdrops, inv, n_inv = {}, [], [], 0, 0
    top_inv = top_n = 0
    for t in tabs:
        s = score(t, gamma, c); s.index = t["teams"]; y, wk, g = t["year"], t["week"], t["g"]
        rk = s.rank(ascending=False)
        for r in g[g.week <= wk].itertuples():
            if r.homeTeam in s.index and r.awayTeam in s.index and r.homePoints != r.awayPoints:
                w_, l_ = (r.homeTeam, r.awayTeam) if r.homePoints > r.awayPoints else (r.awayTeam, r.homeTeam)
                n_inv += 1; inv += float(s[w_] > s[l_])
                if rk[w_] <= 25 and rk[l_] <= 25:
                    top_n += 1; top_inv += float(s[w_] > s[l_])
        if (y, wk - 1) in prev:
            b4 = prev[(y, wk - 1)]
            for r in g[g.week == wk].itertuples():
                for tm, me, op in ((r.homeTeam, r.homePoints, r.awayPoints), (r.awayTeam, r.awayPoints, r.homePoints)):
                    if tm in s.index and b4.rank(ascending=False)[tm] <= 25:
                        (drops if me < op else wdrops).append(s[tm] - b4[tm])
        for r in g[g.week == wk + 1].itertuples():
            if r.homeTeam in s.index and r.awayTeam in s.index and r.homePoints != r.awayPoints:
                rowsP.append((y * 100 + wk, float(r.homePoints > r.awayPoints), s[r.homeTeam] - s[r.awayTeam], 0.0 if r.neutralSite else 1.0))
        if y == 2026 and wk in (6, 7):
            cases[wk] = [f"{x}" for x in rk.sort_values().index[:6]] + [int(rk["Alabama"]), int(rk["Kentucky"])]
        prev[(y, wk)] = s
    d = pd.DataFrame(rowsP, columns=["g", "win", "diff", "site"]); X, yy, gr = d[["diff", "site"]].to_numpy(), d.win.to_numpy(), d.g.to_numpy(); ll = 0.0
    for tr, te in GroupKFold(n_splits=10).split(X, yy, gr):
        pr = np.clip(LogisticRegression(C=100, max_iter=2000).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(yy[te] * np.log(pr) + (1 - yy[te]) * np.log(1 - pr)).sum()
    return inv / n_inv, top_inv / top_n, np.mean(drops), np.mean(wdrops), len(drops), ll / len(d), cases


print("gamma  loss-pen  explained(all)  explained(top25 v top25)  predict log-loss   week-7 top 6 (Alabama rank, Kentucky rank)")
for gamma in (0.15, 0.08, 0.0):
    for c in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        a, b, d, w, n, ll, cs = run(gamma, c)
        print(f"{gamma:5.2f}  {c:7.1f}   {a:12.3f}   {b:20.3f}   {ll:14.4f}   {cs[7]}")

# anatomy of a loss under the current score: what the drop depends on
rows = []
prev = {}
for t in tabs:
    s = score(t, 0.15, 0.0); s.index = t["teams"]; y, wk, g = t["year"], t["week"], t["g"]
    if (y, wk - 1) in prev:
        b4 = prev[(y, wk - 1)]; rk = s.rank(ascending=False); rb = b4.rank(ascending=False); Rv = t["R"]
        for r in g[g.week == wk].itertuples():
            for tm, opp, me, op, home in ((r.homeTeam, r.awayTeam, r.homePoints, r.awayPoints, 1), (r.awayTeam, r.homeTeam, r.awayPoints, r.homePoints, 0)):
                if tm in s.index and rb[tm] <= 25 and me < op:
                    rows.append(dict(year=y, week=wk, team=tm, opp=opp, oppR=Rv.get(opp, np.nan), myR=Rv[tm], gap=Rv.get(opp, np.nan) - Rv[tm], margin=op - me, home=home,
                                     was=int(rb[tm]), now=int(rk[tm]), drop=s[tm] - b4[tm], spots=int(rk[tm] - rb[tm])))
    prev[(y, wk)] = s
d = pd.DataFrame(rows).dropna()
print(f"\n{len(d)} losses by teams ranked in the top 25 the week before")
print(f"mean score drop {d['drop'].mean():.3f}, mean rank change {d.spots.mean():.1f} spots; {np.mean(d.spots <= 0) * 100:.0f}% of these teams did not fall at all")
d["opp_strong"] = d.gap > 0
print(d.groupby("opp_strong")[["drop", "spots"]].mean().round(3))
print(d[["drop", "gap", "margin", "home"]].corr().round(2)["drop"])
print(d.sort_values("spots").head(12)[["year", "week", "team", "opp", "margin", "was", "now", "drop"]].to_string(index=False))
