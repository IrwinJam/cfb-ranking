"""Which ratings predict the next week's games best? Walk-forward test on 2025 (whole season) and 2026 so far.

For every week g from 4 on, each model is fit on games through week g and asked about the games of week g+1.
A one-number calibration turns each model's predicted margin into a win probability, so the models are
compared on log-loss, accuracy and how close the margin lands.

  python research/backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr import components as C
from aapr.cfbd_client import CFBD
from aapr.config import INDEPENDENT_TEAMS, TOP_GROUP
from aapr.params_io import load_params
from aapr.ratings import fit_ratings

api = CFBD(ttl_hours=24 * 365)
P = load_params("fitted", quiet=True)
YEARS = {2025: range(1, 17), 2026: range(1, 8)}
FIRST = 4

ep = C.expected_points_curve(api)
raw = {y: C.load_team_games(api, y, w) for y, w in YEARS.items()}
for _y in raw:
    raw[_y]["pts_c"] = raw[_y].pts.clip(upper=56)    # a cap on lopsided scores
games = {y: C.load_games(api, y) for y in YEARS}
drives = {y: pd.DataFrame([r for wk in w for r in api.get("/drives", year=y, week=wk, seasonType="regular", classification="fbs")])
          for y, w in YEARS.items()}
pooled = pd.concat(drives.values(), ignore_index=True)
_, punt_fit, kick_start, fg_model = C.special_teams_events(pooled, ep)
st = {}
for y in YEARS:
    ev, *_ = C.special_teams_events(drives[y], ep, fg_model, punt_fit, kick_start)
    st[y] = C.special_teams_by_team_game(ev)
print(f"punt: expected start = {punt_fit[0]:.1f} + {punt_fit[1]:.2f} x spot; kickoff start {kick_start:.1f}; "
      f"fg make at 30/40/50 yards: {fg_model(30):.2f}/{fg_model(40):.2f}/{fg_model(50):.2f}")


def conf_sets(g):
    c = {}
    for r in g.itertuples():
        c[r.homeTeam] = r.homeConference
        c[r.awayTeam] = r.awayConference
    for t in INDEPENDENT_TEAMS:
        c[t] = INDEPENDENT_TEAMS and "Independent"
    return {t for t, k in c.items() if k in TOP_GROUP}


lines = {}
for y in YEARS:
    ln = pd.DataFrame(api.get("/lines", year=y, seasonType="regular"))
    rows = []
    for r in ln.itertuples():
        sp = [x["spread"] for x in r.lines if x.get("spread") is not None]
        if sp:
            rows.append((r.id, -float(np.mean(sp))))
    lines[y] = pd.DataFrame(rows, columns=["gameId", "market"])


def site(g):
    return np.where(g.neutralSite.astype(bool), 0, 1)


def pred_games(y, wk, ratings_fn):
    """Games of week wk with every model's predicted home margin."""
    gm = games[y][games[y].week == wk]
    return ratings_fn(gm)


def walk(y):
    out = []
    G, TG = games[y], raw[y]
    p4 = conf_sets(G)
    last = int(G.week.max())
    for wk in range(FIRST, last):
        past = G[G.week <= wk]
        nxt = G[G.week == wk + 1].copy()
        if nxt.empty:
            continue
        s = site(nxt)
        rec = nxt[["gameId", "homeTeam", "awayTeam", "homePoints", "awayPoints"]].copy()
        rec["year"], rec["week"] = y, wk + 1
        rec["margin"] = (nxt.homePoints - nxt.awayPoints).to_numpy(float)
        # margin-based rating, as in production but without a preseason prior
        theta, h, _ = fit_ratings(past[["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]], wk, P, None, p4)
        rec["theta"] = [(theta.get(a, np.nan) - theta.get(b, np.nan)) + h * ss for a, b, ss in zip(nxt.homeTeam, nxt.awayTeam, s)]
        from dataclasses import replace
        P2 = replace(P, kappa0=0.4, tau=16.0)          # the same margin model with a light ridge and slow recency decay
        th2, h2, _ = fit_ratings(past[["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]], wk, P2, None, p4)
        rec["theta2"] = [(th2.get(a, np.nan) - th2.get(b, np.nan)) + h2 * ss for a, b, ss in zip(nxt.homeTeam, nxt.awayTeam, s)]
        # component ratings
        tg = TG[TG.week <= wk]
        for name, col, scale in (("pts", "pts_c", 1.0), ("ppa_tot", "o_tppa", 1.0), ("ppa_play", "o_ppa", 70.0), ("sr", "o_sr", 70.0), ("xr", "o_xr", 70.0)):
            O, D, mu, b = C.fit_off_def(tg, col, wk, tau=16.0, kappa0=0.4)
            nr = O + D
            rec[name] = scale * np.array([nr.get(a, np.nan) - nr.get(bb, np.nan) + 2 * b * ss for a, bb, ss in zip(nxt.homeTeam, nxt.awayTeam, s)])
            rec[name + "_O"] = scale * np.array([O.get(a, np.nan) - O.get(bb, np.nan) for a, bb in zip(nxt.homeTeam, nxt.awayTeam)])
            rec[name + "_D"] = scale * np.array([D.get(a, np.nan) - D.get(bb, np.nan) for a, bb in zip(nxt.homeTeam, nxt.awayTeam)])
        stt, g_ = C.fit_special(st[y], past, wk)
        rec["st"] = [stt.get(a, np.nan) - stt.get(bb, np.nan) + g_ * ss for a, bb, ss in zip(nxt.homeTeam, nxt.awayTeam, s)]
        out.append(rec)
    d = pd.concat(out, ignore_index=True)
    return d.merge(lines[y], on="gameId", how="left")


res = pd.concat([walk(y) for y in YEARS], ignore_index=True)
res = res.dropna(subset=["theta", "theta2", "pts", "ppa_tot", "ppa_play", "st"]).reset_index(drop=True)
res = res[res.margin != 0].reset_index(drop=True)
res["win"] = (res.margin > 0).astype(float)
print(f"{len(res)} games tested ({(res.year == 2025).sum()} in 2025, {(res.year == 2026).sum()} in 2026)")


def calibrate(x, y):
    """Best single slope a for P(home wins) = expit(a * x), plus the log-loss it gives."""
    f = lambda a: -(y * np.log(np.clip(expit(a * x), 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - expit(a * x), 1e-9, 1))).mean()
    r = minimize_scalar(f, bounds=(1e-4, 5), method="bounded")
    return r.x, r.fun


def score(name, x, sub=None):
    d = res if sub is None else res[sub]
    xx = d[name].to_numpy(float) if isinstance(name, str) else name
    ok = ~np.isnan(xx)
    y = d.win.to_numpy()[ok]
    xx = xx[ok]
    a, ll = calibrate(xx, y)
    acc = ((xx > 0) == (y > 0.5)).mean()
    mae = np.abs(d.margin.to_numpy()[ok] - xx).mean() if name not in ("theta",) else np.nan
    return ll, acc, ok.sum()


def table(label, sub=None):
    print(f"\n{label}")
    print(f"{'model':34s} {'log-loss':>9s} {'accuracy':>9s} {'games':>6s}")
    for name, desc in (("theta", "margin ridge (current)"), ("theta2", "margin ridge, light ridge"), ("pts", "points scored/allowed"), ("ppa_tot", "PPA, summed per game"),
                       ("ppa_play", "PPA per play"), ("sr", "success rate"), ("xr", "explosiveness"), ("market", "betting line (benchmark)")):
        ll, acc, n = score(name, None, sub)
        print(f"{desc:34s} {ll:9.4f} {acc:9.3f} {n:6d}")
    return


table("all tested games")
table("2025 only", res.year == 2025)
table("2026 only", res.year == 2026)
res.to_pickle(Path(__file__).with_name("backtest_games.pkl"))
print("\nsaved backtest_games.pkl")
