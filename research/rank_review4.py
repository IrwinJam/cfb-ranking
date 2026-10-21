"""Test the two hand-set choices in the board score: which team is the bubble in the resume, and how much weight strength gets.

Score = (1 - beta) * z(resume) + beta * z(strength), beta rising linearly from beta_early (week 1) to beta_late (week 12).
Each variant is judged four ways, over 2025 and 2026:
  predict    log-loss of next week's games between two top-group teams from the score gap
  explain    share of played top-group games won by the better-scored team, and the same among top-25 pairs
  ap         rank correlation with the AP poll released after the same games
Run once to build the tables (about ten minutes), then it evaluates and prints.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr import components as C
from aapr.cfbd_client import CFBD, CFBDProvider
from aapr.config import INDEPENDENT_TEAMS, TOP_GROUP
from aapr.params_io import load_params
from aapr.pipeline import preseason_prior
from aapr.ratings import fit_ratings, resume_scores

api = CFBD(ttl_hours=24 * 365)
prov = CFBDProvider(ttl_hours=24 * 365)
P = load_params("fitted", quiet=True)
COLS = ["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]
BUBBLES = (8, 12, 18, 25, 32, 40)
PKL = Path(__file__).with_name("rank_review4_tabs.pkl")


def z(s):
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def build():
    out = []
    for y in (2025, 2026):
        g = C.load_games(api, y)
        conf = {}
        for r in g.itertuples():
            conf[r.homeTeam], conf[r.awayTeam] = r.homeConference, r.awayConference
        p4 = {t for t, c in conf.items() if c in TOP_GROUP} | set(INDEPENDENT_TEAMS)
        fbs = {t for r in g.itertuples() for t, c in ((r.homeTeam, r.homeClassification), (r.awayTeam, r.awayClassification)) if c == "fbs"}
        theta0 = preseason_prior(prov, y, P, "season", p4)
        prior = C.points_prior(prov, y, P, p4, fbs)
        for wk in range(3, int(g.week.max()) + 1):
            played = g[g.week <= wk]
            theta, h, _ = fit_ratings(played[COLS], wk, P, None, theta0, p4)
            bl = C.blended_strength(theta, h, played, wk, P, prior, p4, None, fbs)
            R, hfa = bl["R"], bl["hfa"]
            teams = sorted(t for t in p4 if t in R.index)
            srt = R.reindex(teams).sort_values(ascending=False)
            res = {}
            for k in BUBBLES:
                res[k] = resume_scores(played, R, hfa, srt.iloc[min(k - 1, len(teams) - 1)], P.sigma_blend, p4, P.nonp4_win_weight, P.group_game_weight).reindex(teams).fillna(0.0)
            out.append(dict(year=y, week=wk, g=g, teams=teams, R=R.reindex(teams), resume=res))
        print("built", y, flush=True)
    pd.to_pickle(out, PKL)
    return out


def polls_for(tabs):
    polls = {}
    for t in tabs:
        key = (t["year"], t["week"])
        if key in polls:
            continue
        for entry in api.get("/rankings", year=t["year"], week=t["week"] + 1, seasonType="regular"):
            for poll in entry.get("polls", []):
                if poll.get("poll") == "AP Top 25":
                    polls[key] = {x["school"]: int(x["rank"]) for x in poll["ranks"]}
    return polls


def evaluate(tabs, polls, bubble, b_early, b_late):
    inv = n_inv = top_inv = top_n = 0
    rows, rho = [], []
    for t in tabs:
        wk = t["week"]
        b = b_early + (b_late - b_early) * min(wk / P.beta_ramp_weeks, 1)
        s = (1 - b) * z(t["resume"][bubble]) + b * z(t["R"])
        s.index = t["teams"]
        rk = s.rank(ascending=False)
        g = t["g"]
        for r in g[g.week <= wk].itertuples():
            if r.homeTeam in s.index and r.awayTeam in s.index and r.homePoints != r.awayPoints:
                w_, l_ = (r.homeTeam, r.awayTeam) if r.homePoints > r.awayPoints else (r.awayTeam, r.homeTeam)
                n_inv += 1; inv += float(s[w_] > s[l_])
                if rk[w_] <= 25 and rk[l_] <= 25:
                    top_n += 1; top_inv += float(s[w_] > s[l_])
        for r in g[g.week == wk + 1].itertuples():
            if r.homeTeam in s.index and r.awayTeam in s.index and r.homePoints != r.awayPoints:
                rows.append((t["year"] * 100 + wk, float(r.homePoints > r.awayPoints), s[r.homeTeam] - s[r.awayTeam], 0.0 if r.neutralSite else 1.0))
        ap = polls.get((t["year"], wk))
        if ap and wk >= 4:
            inap = [x for x in ap if x in s.index]
            if len(inap) >= 12:
                rho.append(spearmanr(s[inap], [-ap[x] for x in inap]).statistic)
    d = pd.DataFrame(rows, columns=["g", "win", "diff", "site"])
    X, y, gr = d[["diff", "site"]].to_numpy(), d.win.to_numpy(), d.g.to_numpy()
    ll = 0.0
    for tr, te in GroupKFold(n_splits=10).split(X, y, gr):
        pr = np.clip(LogisticRegression(C=100, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr)).sum()
    return ll / len(d), inv / n_inv, top_inv / top_n, float(np.mean(rho))


if __name__ == "__main__":
    tabs = pd.read_pickle(PKL) if PKL.exists() else build()
    polls = polls_for(tabs)
    print("Which team is the bubble (strength 0.30 -> 0.50, as now)")
    print(f"{'bubble':>8s}{'log-loss':>10s}{'explained':>11s}{'top-25 pairs':>14s}{'AP corr':>9s}")
    for k in BUBBLES:
        a = evaluate(tabs, polls, k, P.beta_early, 0.5)
        print(f"{'#' + str(k):>8s}{a[0]:10.4f}{a[1]:11.3f}{a[2]:14.3f}{a[3]:9.3f}")
    print("\nHow much weight strength gets by week 12 (bubble #25, early weight 0.30)")
    print(f"{'late':>8s}{'log-loss':>10s}{'explained':>11s}{'top-25 pairs':>14s}{'AP corr':>9s}")
    for late in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        a = evaluate(tabs, polls, 25, P.beta_early, late)
        print(f"{late:8.1f}{a[0]:10.4f}{a[1]:11.3f}{a[2]:14.3f}{a[3]:9.3f}")
    print("\nEarly and late weight together (bubble #25)")
    print(f"{'early/late':>12s}{'log-loss':>10s}{'explained':>11s}{'top-25 pairs':>14s}{'AP corr':>9s}")
    for e in (0.0, 0.15, 0.3, 0.5):
        for late in (0.5, 0.7):
            a = evaluate(tabs, polls, 25, e, late)
            print(f"{e:6.2f}/{late:.2f}{a[0]:12.4f}{a[1]:11.3f}{a[2]:14.3f}{a[3]:9.3f}")
