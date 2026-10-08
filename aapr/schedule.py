"""Strength of schedule, with every opponent valued at the current week's ratings.

For each team: the average opponent rating over games played, games remaining and the full schedule
(road opponents count as stronger), the rank of each among the top group, and how much the played
opponents have moved since those games were played.
"""
import numpy as np
import pandas as pd


def schedule_table(games: pd.DataFrame, theta_now: pd.Series, theta_hist: dict, week: int,
                   rank_now: pd.Series, group: set, h: float) -> pd.DataFrame:
    """games: every scheduled and played game (week, neutralSite, homeTeam, homePoints, awayTeam, awayPoints).
    theta_hist: {week: ratings fitted through that week}. rank_now: team -> rank in the top group."""
    fallback = float(np.percentile(theta_now.to_numpy(), 5))
    rows = []
    for t in sorted(group):
        gm = games[(games.homeTeam == t) | (games.awayTeam == t)].sort_values("week")
        home = (gm.homeTeam == t).to_numpy()
        opp = np.where(home, gm.awayTeam, gm.homeTeam)
        site = np.where(gm.neutralSite.astype(bool), 0, np.where(home, 1, -1))
        pf = np.where(home, gm.homePoints, gm.awayPoints).astype(float)
        pa = np.where(home, gm.awayPoints, gm.homePoints).astype(float)
        wk = gm.week.to_numpy()
        done = (wk <= week) & ~np.isnan(pf) & ~np.isnan(pa)
        left = wk > week
        now = np.array([theta_now.get(o, fallback) for o in opp], float) - site * h   # road opponent plays tougher
        then = np.array([theta_hist.get(int(w), theta_now).get(o, fallback) for w, o in zip(wk, opp)], float) - site * h
        mean = lambda m, v=now: float(v[m].mean()) if m.any() else np.nan
        wins, losses = done & (pf > pa), done & (pf < pa)
        top25 = np.array([rank_now.get(o, 999) <= 25 for o in opp])
        win_ranks = [(rank_now.get(o, 999), o) for o in opp[wins]]
        best = min(win_ranks) if win_ranks else None
        loss_ranks = [(rank_now.get(o, 999), o) for o in opp[losses]]
        worst = max(loss_ranks) if loss_ranks else None
        rows.append(dict(
            team=t, n_played=int(done.sum()), n_remaining=int(left.sum()),
            sos_played=mean(done), sos_remaining=mean(left), sos_all=mean(done | left),
            revision=mean(done) - mean(done, then) if done.any() else np.nan,
            top25_wins=int((wins & top25).sum()), top25_losses=int((losses & top25).sum()),
            best_win=(f"{best[1]} (#{best[0]})" if best and best[0] < 999 else (best[1] if best else "")),
            worst_loss=(f"{worst[1]} (#{worst[0]})" if worst and worst[0] < 999 else (worst[1] if worst else ""))))
    out = pd.DataFrame(rows)
    for c in ("sos_played", "sos_remaining", "sos_all"):
        out[c + "_rank"] = out[c].rank(ascending=False, method="min")
    return out
