"""Fit the model's settings to game results.

    python fit_to_games.py --year 2026 --through 5

Ratings built from games through week g predict the games of week g+1. The score is log-loss on games
involving at least one top-group team, so a setting only earns credit for games it has not seen.
Two settings stay fixed: the weight on wins over outside teams is 0, and the blend between rating and
resume is 0.5. A leave-one-week-out run compares the fit with the defaults and with the AP-fitted
settings. Output goes to fitted_params.json and outputs/<year>/game_fit_history.csv.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from aapr.cfbd_client import CFBDProvider
from aapr.config import INDEPENDENT_TEAMS, TOP_GROUP, Params
from aapr.pipeline import preseason_prior
from aapr.ratings import fit_ratings
from fit_to_ap import Memo

SPEC = {  # name: (default, low, high)
    "group_game_weight": (1.5, 1.0, 3.0),
    "k": (4.0, 2.0, 8.0),
    "tau": (6.0, 2.0, 20.0),
    "kappa0": (3.0, 0.5, 10.0),
    "prior_shrink": (0.6, 0.0, 1.5),
    "hfa_scale": (1.0, 0.0, 2.0),
    "upset_bonus": (0.0, 0.0, 1.0),
    "sigma_g": (11.0, 6.0, 18.0),
}
NAMES = list(SPEC)
PINNED = dict(nonp4_win_weight=0.0, beta_early=0.3, beta_late=0.5)
REG = 0.02


def make_params(v):
    return replace(Params(), **PINNED, **dict(zip(NAMES, v)))


def top_group(prov, year):
    teams = prov.teams(year)
    conf = dict(zip(teams.school, teams.conference))
    for t in INDEPENDENT_TEAMS:
        conf[t] = "Independent"
    return {t for t, c in conf.items() if c in TOP_GROUP}


def predict_week(prov, year, gw, params, grp):
    """Predict the games of week gw+1 from ratings through week gw."""
    games = prov.games(year)
    played = games[(games.week <= gw) & games.homePoints.notna() & games.awayPoints.notna()]
    theta0 = preseason_prior(prov, year, params, "season", grp)
    th, h, sd = fit_ratings(played, gw, params, theta0, grp)
    nxt = games[(games.week == gw + 1) & games.homePoints.notna() & games.awayPoints.notna()]
    nxt = nxt[nxt.homeTeam.isin(th.index) & nxt.awayTeam.isin(th.index)
              & (nxt.homeTeam.isin(grp) | nxt.awayTeam.isin(grp))]
    site = (~nxt.neutralSite.astype(bool)).astype(float).to_numpy()
    z = (th[nxt.homeTeam].to_numpy() - th[nxt.awayTeam].to_numpy() + site * h) / np.sqrt(
        sd[nxt.homeTeam].to_numpy() ** 2 + sd[nxt.awayTeam].to_numpy() ** 2 + params.sigma_g ** 2)
    y = np.where(nxt.homePoints > nxt.awayPoints, 1.0, np.where(nxt.homePoints < nxt.awayPoints, 0.0, 0.5))
    return pd.DataFrame(dict(p=np.clip(norm.cdf(z), 0.02, 0.98), y=y,
                             both=(nxt.homeTeam.isin(grp) & nxt.awayTeam.isin(grp)).to_numpy()))


def logloss(d):
    return float(-np.mean(d.y * np.log(d.p) + (1 - d.y) * np.log(1 - d.p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--through", type=int)
    ap.add_argument("--maxiter", type=int, default=3)
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()

    prov = Memo(CFBDProvider())
    g = prov.games(a.year)
    full = g.groupby("week").homePoints.apply(lambda s: s.notna().mean())
    through = a.through or int(max(w for w in full.index if full[w] >= 0.9))
    grp = top_group(prov, a.year)
    pred_weeks = list(range(1, through))            # ratings through w predict week w+1
    defaults = np.array([SPEC[n][0] for n in NAMES])
    lo = np.array([SPEC[n][1] for n in NAMES])
    hi = np.array([SPEC[n][2] for n in NAMES])
    cache = {}

    def preds(v, w):
        key = (tuple(np.round(v, 4)), w)
        if key not in cache:
            cache[key] = predict_week(prov, a.year, w, make_params(v), grp)
        return cache[key]

    def score(v, weeks, reg=True):
        d = pd.concat([preds(v, w) for w in weeks])
        r = REG * float(np.mean(((v - defaults) / (hi - lo)) ** 2)) if reg else 0.0
        return logloss(d) + r

    def fit(weeks, x0, maxiter):
        res = minimize(lambda v: score(v, weeks), x0, method="Powell", bounds=list(zip(lo, hi)),
                       options=dict(maxiter=maxiter, xtol=1e-2, ftol=1e-4))
        return np.clip(res.x, lo, hi)

    t0 = time.time()
    n_games = sum(len(preds(defaults, w)) for w in pred_weeks)
    base = score(defaults, pred_weeks, reg=False)
    best = fit(pred_weeks, defaults, a.maxiter)
    fit_loss = score(best, pred_weeks, reg=False)
    d_best = pd.concat([preds(best, w) for w in pred_weeks])
    print(f"fitted on {n_games} games (weeks 2-{through}) in {time.time() - t0:.0f}s: "
          f"log-loss {base:.4f} -> {fit_loss:.4f}  (coin flip = 0.6931)")
    print(f"  top-group vs top-group games only: {logloss(d_best[d_best.both]):.4f} "
          f"({int(d_best.both.sum())} games)")

    ap_file = Path("fitted_params_ap.json")
    ap_loss = None
    if ap_file.exists():
        apv = json.loads(ap_file.read_text())["params"]
        v_ap = np.array([apv.get(n, SPEC[n][0]) for n in NAMES])
        ap_loss = score(v_ap, pred_weeks, reg=False)
        print(f"  earlier AP-fitted settings on the same games: {ap_loss:.4f}")

    print("\nsetting            default   fitted   loss range over its bounds (bigger = better determined)")
    for i, n in enumerate(NAMES):
        ls = []
        for val in np.linspace(lo[i], hi[i], 5):
            v = best.copy()
            v[i] = val
            ls.append(score(v, pred_weeks, reg=False))
        print(f"{n:18s} {defaults[i]:7.2f} {best[i]:8.2f}   {max(ls) - min(ls):8.4f}")

    cv = None
    if not a.no_cv and len(pred_weeks) >= 3:
        folds = []
        for w in pred_weeks:
            train = [x for x in pred_weeks if x != w]
            v = fit(train, best, 2)
            folds.append((w, logloss(preds(v, w)), logloss(preds(defaults, w)),
                          logloss(preds(v_ap, w)) if ap_loss is not None else np.nan))
        f = pd.DataFrame(folds, columns=["held_out_week", "fitted", "defaults", "ap_fitted"])
        cv = f
        print("\nleave-one-week-out log-loss (settings refit without that week):")
        print(f.round(4).to_string(index=False))
        print(f"mean: fitted {f.fitted.mean():.4f} | defaults {f.defaults.mean():.4f} | AP-fitted {f.ap_fitted.mean():.4f}")
        cv_summary = dict(through_week=through, n_games=int(n_games), weeks=int(len(f)), fitted=round(float(f.fitted.mean()), 4),
                          defaults=round(float(f.defaults.mean()), 4), ap_fitted=round(float(f.ap_fitted.mean()), 4))
        cv_dir = Path("outputs") / str(a.year)
        cv_dir.mkdir(parents=True, exist_ok=True)
        (cv_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2))

    vals = {n: round(float(best[i]), 4) for i, n in enumerate(NAMES)}
    allp = dict(group_game_weight=vals["group_game_weight"], nonp4_win_weight=0.0, k=vals["k"], tau=vals["tau"],
                kappa0=vals["kappa0"], prior_shrink=vals["prior_shrink"], hfa_scale=vals["hfa_scale"],
                upset_bonus=vals["upset_bonus"], beta_late=0.5, sigma_g=vals["sigma_g"])
    cur = Path("fitted_params.json")
    if cur.exists() and not ap_file.exists():
        ap_file.write_text(cur.read_text())            # keep the AP fit
    cur.write_text(json.dumps(dict(through_week=through, year=a.year, source="game results", params=allp,
                                   logloss_default=round(base, 4), logloss_fit=round(fit_loss, 4)), indent=2))
    out = Path("outputs") / str(a.year)
    out.mkdir(parents=True, exist_ok=True)
    row = dict(through_week=through, fitted_at=time.strftime("%Y-%m-%d %H:%M"), n_games=n_games,
               logloss_default=round(base, 4), logloss_fit=round(fit_loss, 4),
               cv_logloss_fit=round(float(cv.fitted.mean()), 4) if cv is not None else np.nan, **vals)
    hist = out / "game_fit_history.csv"
    df = pd.DataFrame([row])
    if hist.exists():
        old = pd.read_csv(hist)
        df = pd.concat([old[old.through_week != through], df], ignore_index=True).sort_values("through_week")
    df.to_csv(hist, index=False)
    print(f"\nsaved fitted_params.json (AP fit kept in fitted_params_ap.json) and {hist}")


if __name__ == "__main__":
    main()
