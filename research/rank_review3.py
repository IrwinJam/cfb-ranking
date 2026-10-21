"""Which way of combining resume, strength, schedule and record best matches the human consensus (the AP poll)?

The AP poll is not the truth, but it is what the board is read against, and it is the only yardstick for a ranking of who
has earned a place (game results cannot score that). For each week the score is compared with the poll released after
that week's games. Weights are fitted on 2025 only and checked on 2026.  Reads rank_review_tabs.pkl (from rank_review.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr.cfbd_client import CFBD
from aapr.params_io import load_params

P = load_params("fitted", quiet=True)
api = CFBD(ttl_hours=24 * 365)
tabs = pd.read_pickle(Path(__file__).with_name("rank_review_tabs.pkl"))


def z(s):
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


cols = ["resume", "strength", "sos", "record", "losses"]
polls = {}
for y in (2025, 2026):
    for wk in sorted({t["week"] for t in tabs if t["year"] == y}):
        for entry in api.get("/rankings", year=y, week=wk + 1, seasonType="regular"):
            for poll in entry.get("polls", []):
                if poll.get("poll") == "AP Top 25":
                    polls[(y, wk)] = {x["school"]: (int(x["rank"]), float(x.get("points") or 0)) for x in poll["ranks"]}


def losses(t):
    g = t["g"]; g = g[g.week <= t["week"]]
    L = pd.concat([g.assign(t=g.homeTeam, l=g.homePoints < g.awayPoints), g.assign(t=g.awayTeam, l=g.awayPoints < g.homePoints)]).groupby("t")["l"].sum()
    return L.reindex(t["teams"]).fillna(0.0)


def feats(t):
    """z-scored ingredients for one week, one row per top-group team."""
    L = losses(t)
    f = pd.DataFrame({"resume": z(t["resume"]), "strength": z(t["R"]), "sos": z(t["sos"]), "record": z(t["rec"]), "losses": z(L)}, index=t["teams"])
    f["resume_x"] = z(t["resume"] - 2.0 * L)         # resume with a flat loss penalty
    return f


def beta(wk):
    return P.beta_early + (P.beta_late - P.beta_early) * min(wk / P.beta_ramp_weeks, 1)


VARIANTS = {
    "current board score": lambda f, wk: 0.85 * ((1 - beta(wk)) * f.resume + beta(wk) * f.strength) + 0.15 * f.sos,
    "current + loss penalty 2": lambda f, wk: 0.85 * ((1 - beta(wk)) * f.resume_x + beta(wk) * f.strength) + 0.15 * f.sos,
    "no schedule term": lambda f, wk: (1 - beta(wk)) * f.resume + beta(wk) * f.strength,
    "no schedule term + loss penalty 2": lambda f, wk: (1 - beta(wk)) * f.resume_x + beta(wk) * f.strength,
    "resume + strength 50/50 + penalty 2": lambda f, wk: 0.5 * f.resume_x + 0.5 * f.strength,
    "resume 60 / strength 40": lambda f, wk: 0.6 * f.resume + 0.4 * f.strength,
    "resume 40 / strength 60": lambda f, wk: 0.4 * f.resume + 0.6 * f.strength,
    "resume only": lambda f, wk: f.resume,
    "strength only": lambda f, wk: f.strength,
    "record only (wins minus losses)": lambda f, wk: f.record,
    "half resume, half record": lambda f, wk: 0.5 * f.resume + 0.5 * f.record,
    "resume + strength 50/50": lambda f, wk: 0.5 * f.resume + 0.5 * f.strength,
}


def agree(fn, years, wmin=4):
    rho, over, rows = [], [], []
    for t in tabs:
        if t["year"] not in years or t["week"] < wmin:
            continue
        ap = polls.get((t["year"], t["week"]))
        if not ap:
            continue
        f = feats(t)
        s = fn(f, t["week"]) if callable(fn) else f[cols].values @ fn
        s = pd.Series(np.asarray(s), index=t["teams"])
        inap = [x for x in ap if x in s.index]
        if len(inap) < 12:
            continue
        rho.append(spearmanr(s[inap], [-ap[x][0] for x in inap]).statistic)
        top_ap = {x for x in inap if ap[x][0] <= 10}
        over.append(len(top_ap & set(s.sort_values(ascending=False).index[:10])) / max(len(top_ap), 1))
    return np.mean(rho), np.mean(over)


print(f"{'variant':36s}{'2025 rank corr':>15s}{'top-10 overlap':>16s}{'2026 rank corr':>16s}{'top-10 overlap':>16s}")
for name, fn in VARIANTS.items():
    a, b = agree(fn, {2025}), agree(fn, {2026})
    print(f"{name:36s}{a[0]:15.3f}{a[1]:16.2f}{b[0]:16.3f}{b[1]:16.2f}")

# weights fitted to the poll on 2025, checked on 2026
cols = ["resume", "strength", "sos", "record", "losses"]
X, Y = [], []
for t in tabs:
    ap = polls.get((t["year"], t["week"]))
    if t["year"] != 2025 or t["week"] < 4 or not ap:
        continue
    f = feats(t)[cols]
    y = np.array([max(0, 26 - ap[x][0]) if x in ap else 0 for x in f.index], float)
    X.append(f.values); Y.append(y / 25)
X, Y = np.vstack(X), np.concatenate(Y)
w = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)[0]
print("\nweights fitted to the 2025 polls (resume, strength, schedule, record, losses):", np.round(w[:-1] / np.abs(w[:-1]).sum(), 2))
fw = w[:-1]
a, b = agree(fw, {2025}), agree(fw, {2026})
print(f"{'fitted to 2025':36s}{a[0]:15.3f}{a[1]:16.2f}{b[0]:16.3f}{b[1]:16.2f}")

# week 7 of 2026 under each variant
t = [t for t in tabs if t["year"] == 2026 and t["week"] == 7][0]
f = feats(t)
print("\nweek 7 of 2026, top 10 under each variant")
for name, fn in VARIANTS.items():
    s = fn(f, 7).sort_values(ascending=False)
    print(f"  {name:34s} {', '.join(s.index[:10])}   TTU #{list(s.index).index('Texas Tech') + 1}  Alabama #{list(s.index).index('Alabama') + 1}")
s = pd.Series(f[cols].values @ fw, index=f.index).sort_values(ascending=False)
print(f"  {'fitted to 2025':34s} {', '.join(s.index[:10])}   TTU #{list(s.index).index('Texas Tech') + 1}  Alabama #{list(s.index).index('Alabama') + 1}")
print("  AP poll after week 7:", ", ".join(sorted(polls[(2026, 7)], key=lambda x: polls[(2026, 7)][x][0])[:12]))
