"""Opponent-adjusted ridge ratings.

Each game gives one equation: compress(margin) = rating_home - rating_away + home_field.
All teams are solved together by weighted ridge regression toward a preseason prior.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Params


def compress(margin, k):
    return k * np.sign(margin) * np.log1p(np.abs(margin))


def fit_ratings(games: pd.DataFrame, now_week: int, params: Params,
                theta0: pd.Series | None = None, p4: set | None = None):
    """games: week, neutralSite, homeTeam, homePoints, awayTeam, awayPoints (played games only).
    theta0: preseason prior by team. p4: the top group, used by the stage-2 rules below.

    Stage 1 rates every team. Stage 2 re-solves the top-group ratings with every other rating held
    fixed, ignoring top-group wins over outside teams and keeping their losses.
    Returns (ratings with mean 0, home-field edge, standard error of each rating).
    """
    g = games.reset_index(drop=True)
    teams = sorted(set(g.homeTeam) | set(g.awayTeam))
    ix = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(g)
    X = np.zeros((m, n + 1))
    hi = g.homeTeam.map(ix).to_numpy()
    ai = g.awayTeam.map(ix).to_numpy()
    X[np.arange(m), hi] = 1
    X[np.arange(m), ai] = -1
    X[:, n] = (~g.neutralSite.astype(bool)).astype(float)
    margin = (g.homePoints - g.awayPoints).to_numpy(float)
    y = compress(margin, params.k)
    w = np.exp(-(now_week - g.week.to_numpy(float)) / params.tau)
    if p4:
        in4 = np.array([t in p4 for t in teams])
        w = np.where(in4[hi] & in4[ai], w * params.group_game_weight, w)

    ngames = np.bincount(np.r_[hi, ai], minlength=n)
    kappa = params.kappa0 / (1 + ngames / params.n0)
    prior = np.zeros(n) if theta0 is None else theta0.reindex(teams).fillna(0.0).to_numpy()

    def solve(wt, h_target=None):
        XtW = X.T * wt
        A = XtW @ X + np.diag(np.r_[kappa, 1e-3])
        rhs = XtW @ y + np.r_[kappa * prior, 0.0]
        if h_target is not None:          # pin the home-field edge to a scaled estimate
            A[n, n] += 1e6
            rhs[n] += 1e6 * h_target
        return np.linalg.solve(A, rhs), A

    sol, A = solve(w)
    h_target = params.hfa_scale * sol[n] if params.hfa_scale != 1.0 else None
    if h_target is not None:
        sol, A = solve(w, h_target)
    if params.upset_bonus > 0:            # games won by the lower-rated team get extra weight
        won_home = margin >= 0
        th_ = sol[:n]
        gap = np.where(won_home, th_[ai] - th_[hi], th_[hi] - th_[ai])
        w = w * (1 + params.upset_bonus * np.clip(gap / 5, 0, 2))
        sol, A = solve(w, h_target)
    theta, h = sol[:n].copy(), sol[n]
    resid = y - X @ sol
    s2 = (w * resid ** 2).sum() / max(w.sum() - 5, 1)
    sd = np.sqrt(np.maximum(np.diag(np.linalg.inv(A))[:n], 0) * s2)
    theta = theta - theta.mean()

    if p4 and params.nonp4_win_weight != 1.0:
        is4 = np.array([t in p4 for t in teams])
        hp, ap_ = is4[hi], is4[ai]
        p4_won = (hp & ~ap_ & (margin >= 0)) | (ap_ & ~hp & (margin <= 0))
        w2 = np.where(p4_won, w * params.nonp4_win_weight, w)
        keep = (hp | ap_) & (w2 > 0)
        X2, y2, w2 = X[keep][:, :n], y[keep] - h * X[keep, n], w2[keep]
        y2 = y2 - X2[:, ~is4] @ theta[~is4]          # fixed opponents move to the right-hand side
        X2 = X2[:, is4]
        ng2 = np.bincount(np.r_[hi[keep][hp[keep]], ai[keep][ap_[keep]]], minlength=n)[is4]
        k2 = params.kappa0 / (1 + ng2 / params.n0)
        XtW2 = X2.T * w2
        A2 = XtW2 @ X2 + np.diag(k2)
        th2 = np.linalg.solve(A2, XtW2 @ y2 + k2 * prior[is4])
        s22 = (w2 * (y2 - X2 @ th2) ** 2).sum() / max(w2.sum() - 3, 1)
        theta[is4] = th2
        sd = sd.copy()
        sd[is4] = np.sqrt(np.maximum(np.diag(np.linalg.inv(A2)), 0) * s22)
    return pd.Series(theta, teams), float(h), pd.Series(sd, teams)


def win_prob(p_team, p_opp, site, h, var_team, var_opp, sigma_g):
    """site is +1 for a home team, -1 for an away team, 0 at a neutral site."""
    z = (p_team - p_opp + site * h) / np.sqrt(var_team + var_opp + sigma_g ** 2)
    return norm.cdf(z)


def resume_scores(played: pd.DataFrame, theta: pd.Series, h: float, bubble_theta: float,
                  sigma_g: float, p4: set | None = None, nonp4_win_weight: float = 1.0,
                  group_weight: float = 1.0, loss_penalty: float = 0.0) -> pd.Series:
    """Wins above a bubble team: per game, the result minus the chance the bubble team would have won, less loss_penalty per loss.

    Without the penalty a loss to a strong team is nearly free, since a bubble team would usually lose that game too."""
    rows = []
    for r in played.itertuples():
        hs = 0.0 if r.neutralSite else 1.0
        res = 0.5 if r.homePoints == r.awayPoints else float(r.homePoints > r.awayPoints)
        pb_h = norm.cdf((bubble_theta - theta[r.awayTeam] + hs * h) / sigma_g)
        pb_a = norm.cdf((bubble_theta - theta[r.homeTeam] - hs * h) / sigma_g)
        wh = wa = 1.0
        if p4:
            if r.homeTeam in p4 and r.awayTeam not in p4 and res > 0:
                wh = nonp4_win_weight
            if r.awayTeam in p4 and r.homeTeam not in p4 and res < 1:
                wa = nonp4_win_weight
        gw = group_weight if p4 and r.homeTeam in p4 and r.awayTeam in p4 else 1.0
        rows.append((r.homeTeam, gw * wh * (res - pb_h) - loss_penalty * (1 - res)))
        rows.append((r.awayTeam, gw * wa * ((1 - res) - pb_a) - loss_penalty * res))
    return pd.DataFrame(rows, columns=["team", "r"]).groupby("team")["r"].sum()
