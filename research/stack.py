"""Grouped cross-validation (by week) of which ratings add information, pooled over 2025 and 2026."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr.cfbd_client import CFBD

api = CFBD(ttl_hours=24 * 365)
res = pd.read_pickle(Path(__file__).with_name("backtest_games.pkl"))
tal = {(y, r["team"]): float(r["talent"]) for y in (2025, 2026) for r in api.get("/talent", year=y)}
res["talent"] = [(tal.get((y, a), np.nan) - tal.get((y, b), np.nan)) / 100.0 for y, a, b in zip(res.year, res.homeTeam, res.awayTeam)]
res["has_talent"] = res.talent.notna().astype(float)
res["talent"] = res.talent.fillna(0.0)
res = res.dropna(subset=["theta", "pts", "ppa_tot", "ppa_play", "st"]).reset_index(drop=True)
res["grp"] = res.year * 100 + res.week
print(len(res), "games")


def cv(features, label=None, C=1.0, data=None):
    d = res if data is None else data
    X, y, g = d[features].to_numpy(float), d.win.to_numpy(), d.grp.to_numpy()
    ll, acc, n = 0.0, 0, 0
    for tr, te in GroupKFold(n_splits=10).split(X, y, g):
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum()
        acc += ((p > 0.5) == (y[te] > 0.5)).sum()
        n += len(te)
    print(f"{(label or '+'.join(features)):78s} {ll / n:.4f}  {acc / n:.3f}")
    return ll / n


print(f"{'features':78s} logloss  acc")
for f in (["theta"], ["pts"], ["ppa_tot"], ["theta", "pts"], ["theta", "ppa_tot"], ["theta", "st"], ["theta", "talent"],
          ["theta", "pts", "ppa_tot"], ["theta", "pts", "ppa_tot", "st"], ["theta", "pts", "ppa_tot", "st", "talent"],
          ["theta", "pts_O", "pts_D"], ["theta", "ppa_tot_O", "ppa_tot_D"], ["theta", "pts_O", "pts_D", "st"],
          ["theta", "pts_O", "pts_D", "ppa_tot_O", "ppa_tot_D", "st"], ["theta", "pts_O", "pts_D", "ppa_tot_O", "ppa_tot_D", "st", "talent"],
          ["theta", "pts_O", "pts_D", "ppa_play_O", "ppa_play_D", "sr_O", "sr_D", "st", "talent"],
          ["pts_O", "pts_D", "ppa_tot_O", "ppa_tot_D", "st"]):
    cv(f)

print("\nfit on everything, standardized coefficients")
F = ["theta", "pts_O", "pts_D", "ppa_tot_O", "ppa_tot_D", "st", "talent"]
X = res[F].to_numpy(float)
sd = X.std(0)
m = LogisticRegression(C=1.0, max_iter=2000).fit(X / sd, res.win)
for f, c in zip(F, m.coef_[0]):
    print(f"  {f:12s} {c:+.3f}")
d = res.dropna(subset=["market"])
print(f"\ngames with a line: {len(d)}")
cv(["theta"], "theta (games with a line)", data=d)
cv(["market"], "betting line", data=d)
cv(["theta", "pts_O", "pts_D", "ppa_tot_O", "ppa_tot_D", "st", "talent"], "theta + components + talent (games with a line)", data=d)
cv(["market", "theta", "pts_O", "pts_D", "ppa_tot_O", "ppa_tot_D", "st"], "line + all the ratings", data=d)
