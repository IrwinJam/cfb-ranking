"""Review of how the board score is built: does it punish losses, and does it predict or explain results well?

For every week of 2025 and 2026 it rebuilds the blended strength, the resume (wins above bubble) and the
strength of schedule played, then scores variants of the final board score three ways:
  predict   log-loss of next week's games between two top-group teams from the score gap (10-fold, grouped by week)
  explain   share of already-played top-group games won by the better-scored team (fewer inversions is better)
  lossdrop  mean change in score (in z units) the week after a loss, against the week after a win
"""
import sys
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
from aapr.ratings import fit_ratings, resume_scores
from aapr.schedule import schedule_table

api = CFBD(ttl_hours=24 * 365)
prov = CFBDProvider(ttl_hours=24 * 365)
P = load_params("fitted", quiet=True)
COLS = ["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]


def z(s):
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def weekly_tables():
    """One dict per (year, week): games played so far, and per-team strength, resume, sos, record."""
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
            bubble = R.reindex(teams).sort_values(ascending=False).iloc[min(24, len(teams) - 1)]
            res = resume_scores(played, R, hfa, bubble, P.sigma_blend, p4, P.nonp4_win_weight, P.group_game_weight)
            sos = schedule_table(played, R, {}, wk, pd.Series(dtype=float), set(teams), hfa).set_index("team")["sos_played"]
            wins = pd.concat([played.assign(t=played.homeTeam, w=played.homePoints > played.awayPoints),
                              played.assign(t=played.awayTeam, w=played.awayPoints > played.homePoints)]).groupby("t")["w"].agg(["sum", "count"])
            rec = (wins["sum"] - (wins["count"] - wins["sum"])).reindex(teams).fillna(0.0)
            out.append(dict(year=y, week=wk, g=g, p4=p4, teams=teams, R=R.reindex(teams), hfa=hfa,
                            resume=res.reindex(teams).fillna(0.0), sos=sos.reindex(teams).fillna(sos.min()), rec=rec))
        print("built", y, flush=True)
    return out


def score(t, gamma=0.15, rho=0.0, beta=None, drop_resume=False):
    wk = t["week"]
    b = beta if beta is not None else P.beta_early + (P.beta_late - P.beta_early) * min(wk / P.beta_ramp_weeks, 1)
    zR, zP, zS, zW = z(t["resume"]), z(t["R"]), z(t["sos"]), z(t["rec"])
    base = (1 - b) * zR + b * zP if not drop_resume else zP
    return (1 - rho) * ((1 - gamma) * base + gamma * zS) + rho * zW


VARIANTS = {
    "current (sos .15)": dict(),
    "no schedule term": dict(gamma=0.0),
    "sos .08": dict(gamma=0.08),
    "strength only": dict(drop_resume=True, gamma=0.0),
    "current + record .15": dict(rho=0.15),
    "current + record .30": dict(rho=0.30),
    "no sos + record .15": dict(gamma=0.0, rho=0.15),
    "no sos + record .30": dict(gamma=0.0, rho=0.30),
}


def evaluate(tabs):
    res = {k: dict(rows=[], inv=[0, 0], drops=[], wdrops=[]) for k in VARIANTS}
    prev = {k: {} for k in VARIANTS}
    for t in tabs:
        wk, g, teams = t["week"], t["g"], t["teams"]
        for name, kw in VARIANTS.items():
            s = score(t, **kw)
            s.index = teams
            # explain: played top-group games through this week
            for r in g[(g.week <= wk)].itertuples():
                if r.homeTeam in s.index and r.awayTeam in s.index and r.homePoints != r.awayPoints:
                    win, lose = (r.homeTeam, r.awayTeam) if r.homePoints > r.awayPoints else (r.awayTeam, r.homeTeam)
                    res[name]["inv"][1] += 1
                    res[name]["inv"][0] += float(s[win] > s[lose])
            # predict: next week's games
            for r in g[g.week == wk + 1].itertuples():
                if r.homeTeam in s.index and r.awayTeam in s.index and r.homePoints != r.awayPoints:
                    res[name]["rows"].append((t["year"] * 100 + wk, float(r.homePoints > r.awayPoints), s[r.homeTeam] - s[r.awayTeam], 0.0 if r.neutralSite else 1.0))
            # lossdrop: this week's score against last week's, for teams that played this week
            key = t["year"]
            if wk - 1 in prev[name].get(key, {}):
                before = prev[name][key][wk - 1]
                gm = g[g.week == wk]
                for r in gm.itertuples():
                    for tm, me, op in ((r.homeTeam, r.homePoints, r.awayPoints), (r.awayTeam, r.awayPoints, r.homePoints)):
                        if tm in s.index and tm in before.index and before.rank(ascending=False)[tm] <= 25:
                            d = s[tm] - before[tm]
                            (res[name]["drops"] if me < op else res[name]["wdrops"]).append(d)
            prev[name].setdefault(key, {})[wk] = s
    print(f"{'variant':26s}{'log-loss':>9s}{'acc':>7s}{'results explained':>19s}{'drop after loss':>17s}{'after win':>11s}")
    for name, r in res.items():
        d = pd.DataFrame(r["rows"], columns=["g", "win", "diff", "site"])
        X, y, gr = d[["diff", "site"]].to_numpy(), d.win.to_numpy(), d.g.to_numpy()
        ll = acc = 0.0
        for tr, te in GroupKFold(n_splits=10).split(X, y, gr):
            pr = np.clip(LogisticRegression(C=100, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            ll += -(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr)).sum()
            acc += ((pr > .5) == (y[te] > .5)).sum()
        print(f"{name:26s}{ll / len(d):9.4f}{acc / len(d):7.3f}{r['inv'][0] / r['inv'][1]:19.3f}{np.mean(r['drops']):17.3f}{np.mean(r['wdrops']):11.3f}   (n={len(d)}, losses={len(r['drops'])})")


if __name__ == "__main__":
    tabs = weekly_tables()
    pd.to_pickle(tabs, Path(__file__).with_name("rank_review_tabs.pkl"))
    evaluate(tabs)
