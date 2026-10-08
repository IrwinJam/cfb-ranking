"""Fit the model's settings to the AP poll, for comparison with the game-results fit.

    python fit_to_ap.py --year 2026 --through 5

The AP poll for "week n" comes out after game week n-1, so game week g is matched to AP week g+1. Both
lists are cut down to the same teams before comparing. The loss is the mean squared rank gap over the
union of the two top 25s, capped at rank 30, plus a pull toward the defaults. Output goes to
fitted_params_ap.json and outputs/<year>/ap_fit_history.csv.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from aapr.cfbd_client import CFBDProvider
from aapr.config import Params
from aapr.pipeline import run_week

# name: (default, low, high)
SPEC = {
    "group_game_weight": (1.5, 1.0, 3.0),   # weight of games between two top-group teams
    "nonp4_win_weight": (0.0, 0.0, 1.0),    # weight of a top-group win over a non-Power-Four team
    "k": (4.0, 2.0, 8.0),                   # margin compression
    "tau": (6.0, 2.0, 20.0),                # recency decay (weeks)
    "kappa0": (3.0, 0.5, 10.0),             # strength of the preseason prior
    "prior_shrink": (0.6, 0.0, 1.5),        # carry-over of last season's rating
    "hfa_scale": (1.0, 0.0, 2.0),           # home-field scale (higher: road wins worth more)
    "upset_bonus": (0.0, 0.0, 1.0),         # extra weight on upsets
    "beta_late": (0.5, 0.0, 1.0),           # weight on forward strength vs resume (early = 0.6x this)
}
NAMES = list(SPEC)
CAP = 30
REG = 20.0


class Memo:
    """Cache provider calls in memory so repeated model runs do not re-read the API cache."""

    def __init__(self, prov):
        self.p, self.c = prov, {}

    def __getattr__(self, name):
        f = getattr(self.p, name)

        def call(*a):
            key = (name,) + a
            if key not in self.c:
                self.c[key] = f(*a)
            return self.c[key]
        return call


def make_params(v):
    d = dict(zip(NAMES, v))
    b = d.pop("beta_late")
    return replace(Params(), beta_early=0.6 * b, beta_late=b, **d)


def week_loss(prov, year, gw, params, ap_by_week):
    r = run_week(prov, year, gw, params, "season")["rankings"]
    ap = ap_by_week[gw + 1]
    ap = ap[ap.school.isin(set(r.team))].sort_values("rank")
    apr = {t: i + 1 for i, t in enumerate(ap.school)}                # AP rank among the same teams
    ar = dict(zip(r.team, r["rank"]))
    union = set(apr) | set(r.team[:25])
    x = np.array([min(apr.get(t, CAP), CAP) for t in union], float)
    y = np.array([min(ar[t], CAP) for t in union], float)
    common = [t for t in apr if t in ar]
    rho = spearmanr([apr[t] for t in common], [ar[t] for t in common])[0] if len(common) > 3 else np.nan
    return float(np.mean((x - y) ** 2)), rho, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--through", type=int, help="last game week to fit (default: latest completed)")
    ap.add_argument("--maxiter", type=int, default=4)
    a = ap.parse_args()

    prov = Memo(CFBDProvider())
    g = prov.games(a.year)
    full = g.groupby("week").homePoints.apply(lambda s: s.notna().mean())
    through = a.through or int(max(w for w in full.index if full[w] >= 0.9))
    ap_by_week = {w: prov.ap(a.year, w) for w in range(2, through + 2)}
    weeks = list(range(1, through + 1))
    defaults = np.array([SPEC[n][0] for n in NAMES])
    lo = np.array([SPEC[n][1] for n in NAMES])
    hi = np.array([SPEC[n][2] for n in NAMES])
    cache = {}

    def loss(v, with_reg=True):
        key = tuple(np.round(v, 4)) + (with_reg,)
        if key not in cache:
            p = make_params(v)
            ls = [week_loss(prov, a.year, w, p, ap_by_week)[0] for w in weeks]
            reg = REG * float(np.mean(((v - defaults) / (hi - lo)) ** 2)) if with_reg else 0.0
            cache[key] = float(np.mean(ls)) + reg
        return cache[key]

    t0 = time.time()
    base = loss(defaults, with_reg=False)
    res = minimize(loss, defaults, method="Powell", bounds=list(zip(lo, hi)),
                   options=dict(maxiter=a.maxiter, xtol=1e-2, ftol=1e-3))
    best = np.clip(res.x, lo, hi)
    fit_loss = loss(best, with_reg=False)
    print(f"fitted weeks 1-{through} against AP weeks 2-{through + 1}: loss {base:.1f} -> {fit_loss:.1f} "
          f"({len(cache)} model runs, {time.time() - t0:.0f}s)")

    # how well does AP pin each setting down? sweep one at a time around the optimum
    print("\nsetting            default   fitted   loss range over its bounds (bigger = better determined)")
    sens = {}
    for i, n in enumerate(NAMES):
        ls = []
        for val in np.linspace(lo[i], hi[i], 5):
            v = best.copy()
            v[i] = val
            ls.append(loss(v, with_reg=False))
        sens[n] = max(ls) - min(ls)
        print(f"{n:18s} {defaults[i]:7.2f} {best[i]:8.2f}   {sens[n]:6.1f}")

    # last week's remaining disagreements
    p = make_params(best)
    l_last, rho, r = week_loss(prov, a.year, through, p, ap_by_week)
    apx = ap_by_week[through + 1]
    apx = apx[apx.school.isin(set(r.team))].sort_values("rank")
    apr = {t: i + 1 for i, t in enumerate(apx.school)}
    gaps = pd.DataFrame({"team": r.team, "aapr": r["rank"], "ap": [apr.get(t) for t in r.team]})
    gaps = gaps[gaps.ap.notna() | (gaps.aapr <= 25)].assign(
        gap=lambda d: d.ap.fillna(CAP) - d.aapr).sort_values("gap", key=abs, ascending=False)
    print(f"\nweek {through}: Spearman vs AP {rho:.2f}. Biggest remaining gaps (AP rank among the same teams, "
          f"+ = AAPR higher):")
    print(gaps.head(8).to_string(index=False))

    vals = {n: round(float(best[i]), 4) for i, n in enumerate(NAMES)}
    Path("fitted_params_ap.json").write_text(json.dumps(dict(through_week=through, year=a.year, params=vals,
                                                         loss_default=round(base, 2), loss_fit=round(fit_loss, 2)),
                                                    indent=2))
    out = Path("outputs") / str(a.year)
    out.mkdir(parents=True, exist_ok=True)
    row = dict(through_week=through, fitted_at=time.strftime("%Y-%m-%d %H:%M"), loss_default=round(base, 2),
               loss_fit=round(fit_loss, 2), spearman_last_week=round(float(rho), 3), **vals)
    hist = out / "ap_fit_history.csv"
    df = pd.DataFrame([row])
    if hist.exists():
        old = pd.read_csv(hist)
        df = pd.concat([old[old.through_week != through], df], ignore_index=True).sort_values("through_week")
    df.to_csv(hist, index=False)
    print(f"\nsaved fitted_params_ap.json and {hist}")


if __name__ == "__main__":
    main()
