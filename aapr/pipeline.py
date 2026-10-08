"""Weekly ranking: ratings through a week, resume, schedule strength, blended score."""
import numpy as np
import pandas as pd

from .config import INDEPENDENT, INDEPENDENT_TEAMS, TOP_GROUP, Params
from .ratings import fit_ratings, resume_scores, win_prob
from .schedule import schedule_table


def _z(s):
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def preseason_prior(provider, year, params, mode="season", p4=None):
    """Last season's ratings, shrunk. Mode "none" starts every team at zero."""
    if mode == "none":
        return None
    try:
        prev = provider.games(year - 1)
        prev = prev[prev.homePoints.notna() & prev.awayPoints.notna()]
        theta, _, _ = fit_ratings(prev, int(prev.week.max()), params, None, p4)
        return params.prior_shrink * theta
    except Exception as e:
        print(f"[warn] no preseason prior ({type(e).__name__}: {e}); using zeros")
        return None


def run_week(provider, year: int, week: int, params: Params = Params(), prior="season") -> dict:
    """Rank the top group using results through `week`."""
    games = provider.games(year)
    teams = provider.teams(year)
    conf = dict(zip(teams.school, teams.conference))
    for t in INDEPENDENT_TEAMS:
        conf[t] = INDEPENDENT
    p4_all = {t for t, c in conf.items() if c in TOP_GROUP}

    played = games[(games.week <= week) & games.homePoints.notna() & games.awayPoints.notna()]
    theta0 = preseason_prior(provider, year, params, prior, p4_all)
    theta, h, sd = fit_ratings(played, week, params, theta0, p4_all)
    var = sd ** 2

    p4 = sorted(t for t, c in conf.items() if c in TOP_GROUP and t in theta.index)
    bubble = theta.reindex(p4).sort_values(ascending=False).iloc[min(24, len(p4) - 1)]
    resume = resume_scores(played, theta, h, bubble, params.sigma_g, p4_all, params.nonp4_win_weight,
                           params.group_game_weight)

    beta = params.beta_early + (params.beta_late - params.beta_early) * min(week / params.beta_ramp_weeks, 1)
    sos = schedule_table(games, theta, {}, week, pd.Series(dtype=float), set(p4), h).set_index("team")["sos_played"]
    zS = _z(sos.reindex(p4).fillna(sos.min()))
    g = params.sos_weight
    score = (1 - g) * ((1 - beta) * _z(resume.reindex(p4)) + beta * _z(theta.reindex(p4))) + g * zS

    nxt = {}
    for r in games[games.week > week].sort_values("week").itertuples():
        site = 0 if r.neutralSite else 1
        nxt.setdefault(r.homeTeam, (r.awayTeam, site))
        nxt.setdefault(r.awayTeam, (r.homeTeam, -site))
    wins = pd.concat([played.assign(t=played.homeTeam, w=played.homePoints > played.awayPoints),
                      played.assign(t=played.awayTeam, w=played.awayPoints > played.homePoints)])
    wl = wins.groupby("t")["w"].agg(["sum", "count"])

    rows = []
    for t in p4:
        o, site = nxt.get(t, (None, 0))
        wp = win_prob(theta[t], theta[o], site, h, var[t], var[o], params.sigma_g) if o in theta.index else np.nan
        rows.append(dict(team=t, conference=conf[t],
                         record=f"{int(wl.loc[t, 'sum'])}-{int(wl.loc[t, 'count'] - wl.loc[t, 'sum'])}",
                         theta=theta[t], resume=resume.get(t, 0.0), score=score[t], next_opp=o, next_win_prob=wp))
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    sch = schedule_table(games, theta, {}, week, out.set_index("team")["rank"], set(p4), h)
    out = out.merge(sch[["team", "sos_played", "sos_played_rank", "sos_remaining", "sos_remaining_rank", "sos_all_rank",
                         "top25_wins", "top25_losses", "best_win"]], on="team", how="left")
    return dict(rankings=out, theta=theta, hfa=h, beta=beta)
