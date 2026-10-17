"""Offense, defense and special teams ratings from play-level efficiency, next to the margin-based rating.

Three pieces, each in points per game against an average opponent:
  offense  how many points of expected value a team's offense adds per game (CollegeFootballData PPA, summed)
  defense  the same for what the team's defense takes away from its opponents
  special  field position from kickoffs, punts and returns, plus field goals against what a kicker of that
           distance should make, converted to points with the expected points of the starting yard line

Each piece is solved with the same recency-weighted ridge idea as the main rating, so a strong offense that
only played weak defenses is marked down.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

OFF_METRICS = {   # column in the advanced-stats table -> what it is
    "ppa_total": "points added, summed over the game",
    "ppa_play": "points added per play",
    "success": "success rate",
    "explosive": "explosiveness",
}


# ---------------------------------------------------------------- loading
def load_team_games(api, year, weeks):
    """One row per team per game: that team's offense numbers, with the game's site and the opponent."""
    rows = [r for wk in weeks for r in api.get("/stats/game/advanced", year=year, week=wk, seasonType="regular", classification="fbs")]
    g = pd.DataFrame(api.get("/games", year=year, seasonType="regular")).rename(columns={"id": "gameId"})
    return build_team_games(rows, g)


def build_team_games(adv_rows, g: pd.DataFrame) -> pd.DataFrame:
    """Join advanced stat rows to the results table (needs gameId, neutralSite, homeTeam, awayTeam, homePoints, awayPoints)."""
    adv = []
    for r in adv_rows:
        o, d = r["offense"], r["defense"]
        adv.append(dict(gameId=r["gameId"], team=r["team"], opp=r["opponent"], week=r["week"],
                        o_plays=o["plays"], o_ppa=o["ppa"], o_tppa=o["totalPPA"], o_sr=o["successRate"], o_xr=o["explosiveness"],
                        o_stuff=o["stuffRate"], d_plays=d["plays"], d_ppa=d["ppa"], d_tppa=d["totalPPA"],
                        d_sr=d["successRate"], d_xr=d["explosiveness"]))
    adv = pd.DataFrame(adv)
    g = g[["gameId", "week", "neutralSite", "homeTeam", "awayTeam", "homePoints", "awayPoints"]]
    g = g[g.homePoints.notna() & g.awayPoints.notna()]
    m = adv.merge(g[["gameId", "neutralSite", "homeTeam", "awayTeam", "homePoints", "awayPoints"]], on="gameId", how="inner")
    m["site"] = np.where(m.neutralSite.astype(bool), 0, np.where(m.team == m.homeTeam, 1, -1))
    m["pts"] = np.where(m.team == m.homeTeam, m.homePoints, m.awayPoints).astype(float)
    m["pts_allowed"] = np.where(m.team == m.homeTeam, m.awayPoints, m.homePoints).astype(float)
    return m.drop(columns=["neutralSite", "homeTeam", "awayTeam", "homePoints", "awayPoints"])


def team_games_from_games(g: pd.DataFrame, cap: float = 56.0) -> pd.DataFrame:
    """Two rows per game (one per offense) built from the plain results table: points scored and the site."""
    rows = []
    for r in g.itertuples():
        site = 0 if r.neutralSite else 1
        gid = getattr(r, "gameId", r.Index)
        rows.append((gid, r.homeTeam, r.awayTeam, r.week, site, float(r.homePoints), float(r.awayPoints)))
        rows.append((gid, r.awayTeam, r.homeTeam, r.week, -site, float(r.awayPoints), float(r.homePoints)))
    out = pd.DataFrame(rows, columns=["gameId", "team", "opp", "week", "site", "pts", "opp_pts"])
    out["pts_c"] = out.pts.clip(upper=cap)         # a cap on lopsided scores
    return out


def load_games(api, year):
    g = pd.DataFrame(api.get("/games", year=year, seasonType="regular"))
    g = g[["id", "week", "neutralSite", "homeTeam", "homeClassification", "homeConference", "awayTeam", "awayClassification", "awayConference", "homePoints", "awayPoints"]]
    return g[g.homePoints.notna() & g.awayPoints.notna()].rename(columns={"id": "gameId"}).reset_index(drop=True)


# ---------------------------------------------------------------- special teams
def expected_points_curve(api):
    """Expected points for a first and ten, by yard line from the team's own goal (1 to 100)."""
    rows = api.get("/ppa/predicted", down=1, distance=10)
    ep = {int(r["yardLine"]): float(r["predictedPoints"]) for r in rows}
    top = max(ep)
    for x in range(top + 1, 100):            # past the table's top, climb toward a touchdown
        ep[x] = ep[top] + (6.2 - ep[top]) * (x - top) / (99 - top)
    ep[100] = 7.0
    ep[0] = ep[1]
    return ep


def _end_to_goal(d: pd.DataFrame) -> pd.Series:
    """Yards to the goal where a drive ended. The feed's own end-spot columns are mirrored for many drives, so use start minus yards."""
    return d.startYardsToGoal - d.yards


def _fg_make_model(drives):
    """Make probability by kick distance, fitted on the field goal drives in `drives`."""
    d = drives[drives.driveResult.isin(["FG", "MISSED FG"])]
    dist = (_end_to_goal(d) + 17).to_numpy(float)
    made = (d.driveResult == "FG").to_numpy(float)
    keep = (dist >= 17) & (dist <= 68)
    x, y = (dist[keep] - 40) / 10, made[keep]

    def nll(b):
        p = np.clip(expit(b[0] + b[1] * x), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    b = minimize(nll, [1.0, -0.5], method="Nelder-Mead").x
    return lambda dd: expit(b[0] + b[1] * (np.asarray(dd, float) - 40) / 10)


def special_teams_events(drives: pd.DataFrame, ep: dict, fg_model=None, punt_fit=None, kick_start=None):
    """Every kickoff, punt and field goal as a value in points, from the kicking side's point of view.

    Returns (events, punt_fit, kick_start). `punt_fit` is the expected start spot after a punt from a given
    spot and `kick_start` the average start after a kickoff, both learned from `drives` unless passed in.
    """
    d = drives.sort_values(["gameId", "driveNumber"]).reset_index(drop=True)
    prev = d.groupby("gameId").shift(1)
    rows = []
    for cur, p in zip(d.itertuples(), prev.itertuples()):
        s = cur.startYardsToGoal
        if pd.isna(p.offense):                       # opening kickoff
            rows.append(("kick", cur.gameId, cur.defense, cur.offense, s, np.nan, cur.plays == 0 and cur.driveResult == "TD"))
        elif cur.startPeriod >= 5:                   # overtime
            continue
        elif p.endPeriod == 2 and cur.startPeriod == 3:     # second half kickoff, whoever scored last
            rows.append(("kick", cur.gameId, cur.defense, cur.offense, s, np.nan, False))
        elif p.offense == cur.offense:
            continue
        elif p.driveResult in ("TD", "FG"):          # kickoff after a score
            rows.append(("kick", cur.gameId, p.offense, cur.offense, s, np.nan, cur.plays == 0 and cur.driveResult == "TD"))
        elif p.driveResult == "PUNT":
            rows.append(("punt", cur.gameId, p.offense, cur.offense, s, p.startYardsToGoal - p.yards, cur.plays == 0 and cur.driveResult == "TD"))
    ev = pd.DataFrame(rows, columns=["kind", "gameId", "kicker", "receiver", "start", "spot", "return_td"])

    if kick_start is None:
        kick_start = float(ev[(ev.kind == "kick") & ~ev.return_td].start.mean())
    if punt_fit is None:
        pu = ev[(ev.kind == "punt") & ~ev.return_td].dropna(subset=["spot"])
        b1, b0 = np.polyfit(pu.spot.to_numpy(float), pu.start.to_numpy(float), 1)
        punt_fit = (float(b0), float(b1))

    def ep_at(start):                                 # value to the team that gets the ball at `start` yards to goal
        return np.array([ep[int(np.clip(100 - x, 0, 100))] for x in start])
    exp_start = np.where(ev.kind == "kick", kick_start, punt_fit[0] + punt_fit[1] * ev.spot.fillna(0))
    got = np.where(ev.return_td, 0.0, ev.start)
    ev["value"] = ep_at(exp_start) - ep_at(got)       # positive when the receiver starts farther back than expected
    ev["team"] = ev.kicker
    ev["opp"] = ev.receiver

    if fg_model is None:
        fg_model = _fg_make_model(d)
    fg = d[d.driveResult.isin(["FG", "MISSED FG"])]
    dist = (_end_to_goal(fg) + 17).to_numpy(float)
    fgv = pd.DataFrame(dict(kind="fg", gameId=fg.gameId.to_numpy(), team=fg.offense.to_numpy(), opp=fg.defense.to_numpy(),
                            value=3.0 * ((fg.driveResult == "FG").to_numpy(float) - fg_model(dist))))
    out = pd.concat([ev[["kind", "gameId", "team", "opp", "value"]], fgv], ignore_index=True)
    return out, punt_fit, kick_start, fg_model


def special_teams_by_team_game(events: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: points of special teams value for that team (zero-sum with its opponent)."""
    a = events[["gameId", "team", "value"]]
    b = events.assign(team=events.opp, value=-events.value)[["gameId", "team", "value"]]
    return pd.concat([a, b]).groupby(["gameId", "team"], as_index=False)["value"].sum().rename(columns={"value": "st"})


# ---------------------------------------------------------------- ridge fits
def _ridge(X, y, w, kappa, prior=None):
    XtW = X.T * w
    A = XtW @ X + np.diag(kappa)
    rhs = XtW @ y + (kappa * prior if prior is not None else 0)
    return np.linalg.solve(A, rhs)


def fit_off_def(tg: pd.DataFrame, metric: str, now_week: int, tau: float = 8.0, kappa0: float = 6.0, n0: float = 4.0,
                weight_col: str | None = None, prior: tuple | None = None):
    """Opponent-adjusted offense and defense from one metric per team-game.

    metric is the column holding the team's offensive number in that game. Model per row (team i vs opp j):
        metric = mu + O_i - D_j + b * site_i
    with D the amount a defense takes away. Returns (O, D, mu, b) with O and D centered on zero.
    """
    g = tg.dropna(subset=[metric]).reset_index(drop=True)
    teams = sorted(set(g.team) | set(g.opp))
    ix = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(g)
    X = np.zeros((m, 2 * n + 2))
    X[np.arange(m), g.team.map(ix)] = 1
    X[np.arange(m), n + g.opp.map(ix)] = -1
    X[:, 2 * n] = 1
    X[:, 2 * n + 1] = g.site
    w = np.exp(-(now_week - g.week.to_numpy(float)) / tau)
    if weight_col:
        w = w * g[weight_col].to_numpy(float)
    games_played = np.bincount(g.team.map(ix), minlength=n)
    kap = kappa0 / (1 + games_played / n0)
    kappa = np.r_[kap, kap, 1e-6, 1e-6]
    pr = None
    if prior is not None:          # (O0, D0): last season's ratings, already shrunk
        pr = np.r_[pd.Series(prior[0]).reindex(teams).fillna(0.0).to_numpy(), pd.Series(prior[1]).reindex(teams).fillna(0.0).to_numpy(), 0.0, 0.0]
    sol = _ridge(X, g[metric].to_numpy(float), w, kappa, pr)
    O, D = sol[:n] - sol[:n].mean(), sol[n:2 * n] - sol[n:2 * n].mean()
    return pd.Series(O, teams), pd.Series(D, teams), float(sol[2 * n]), float(sol[2 * n + 1])


def fit_special(st_games: pd.DataFrame, games: pd.DataFrame, now_week: int, tau: float = 8.0, kappa0: float = 4.0, n0: float = 4.0):
    """Special teams rating per team: ST_home - ST_away (+ a small site term) = special teams margin of the game."""
    g = games[["gameId", "week", "neutralSite", "homeTeam", "awayTeam"]].merge(
        st_games.rename(columns={"team": "homeTeam", "st": "st_home"}), on=["gameId", "homeTeam"], how="left")
    g["st_home"] = g.st_home.fillna(0.0)
    teams = sorted(set(g.homeTeam) | set(g.awayTeam))
    ix = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(g)
    X = np.zeros((m, n + 1))
    X[np.arange(m), g.homeTeam.map(ix)] = 1
    X[np.arange(m), g.awayTeam.map(ix)] = -1
    X[:, n] = (~g.neutralSite.astype(bool)).astype(float)
    w = np.exp(-(now_week - g.week.to_numpy(float)) / tau)
    games_played = np.bincount(np.r_[g.homeTeam.map(ix), g.awayTeam.map(ix)], minlength=n)
    kappa = np.r_[kappa0 / (1 + games_played / n0), 1e-6]
    sol = _ridge(X, g.st_home.to_numpy(float), w, kappa)
    return pd.Series(sol[:n] - sol[:n].mean(), teams), float(sol[n])


def fit_points(tg: pd.DataFrame, now_week: int, tau: float = 16.0, kappa0: float = 6.0, n0: float = 4.0,
               prior: tuple | None = None, group: set | None = None, group_win_weight: float = 0.0,
               col: str = "pts_c", pack: dict | None = None, center_on: set | None = None):
    """Offense and defense ratings from the points each team scored, in two stages like the margin rating.

    Stage one rates every team from every game. Stage two re-solves the ratings of the `group` teams with
    everyone else held fixed, and gives a group team's win over an outside team `group_win_weight` (0 = ignored);
    a loss to an outside team always counts in full.
    Returns (O, D, mu, b), centered on zero over the `center_on` teams (every team by default). `pack`, when given,
    receives what a browser needs to re-solve stage two.
    """
    g = tg.dropna(subset=[col]).reset_index(drop=True)
    teams = sorted(set(g.team) | set(g.opp))
    ix = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(g)
    ti, oi = g.team.map(ix).to_numpy(), g.opp.map(ix).to_numpy()
    y = g[col].to_numpy(float)
    site = g.site.to_numpy(float)
    w = np.exp(-(now_week - g.week.to_numpy(float)) / tau)
    gp = np.bincount(ti, minlength=n)
    kap = kappa0 / (1 + gp / n0)
    O0 = pd.Series(prior[0]).reindex(teams).fillna(0.0).to_numpy() if prior is not None else np.zeros(n)
    D0 = pd.Series(prior[1]).reindex(teams).fillna(0.0).to_numpy() if prior is not None else np.zeros(n)

    X = np.zeros((m, 2 * n + 2))
    X[np.arange(m), ti] = 1
    X[np.arange(m), n + oi] = -1
    X[:, 2 * n] = 1
    X[:, 2 * n + 1] = site
    sol = _ridge(X, y, w, np.r_[kap, kap, 1e-6, 1e-6], np.r_[O0, D0, 0.0, 0.0])
    Ou, Du, mu, b = sol[:n].copy(), sol[n:2 * n].copy(), float(sol[2 * n]), float(sol[2 * n + 1])

    if group:
        isg = np.array([t in group for t in teams])
        tg_, og_ = isg[ti], isg[oi]
        group_won = (tg_ & ~og_ & (g.pts.to_numpy() > g.opp_pts.to_numpy())) | (og_ & ~tg_ & (g.opp_pts.to_numpy() > g.pts.to_numpy()))
        w2 = np.where(group_won, w * group_win_weight, w)
        keep = (tg_ | og_) & (w2 > 0)
        gi = np.cumsum(isg) - 1
        ng = int(isg.sum())
        r = y[keep] - mu - b * site[keep]
        t_in, o_in = tg_[keep], og_[keep]
        r = r + np.where(t_in & ~o_in, Du[oi[keep]], 0.0) - np.where(~t_in & o_in, Ou[ti[keep]], 0.0)
        Xg = np.zeros((int(keep.sum()), 2 * ng))
        rows_ = np.arange(int(keep.sum()))
        Xg[rows_[t_in], gi[ti[keep]][t_in]] = 1
        Xg[rows_[o_in], ng + gi[oi[keep]][o_in]] = -1
        kg = kap[isg]
        solg = _ridge(Xg, r, w2[keep], np.r_[kg, kg], np.r_[O0[isg], D0[isg]])
        # Adding the same amount to every offense and every defense changes no prediction, and with a top-group team's
        # wins over outsiders ignored only the prior pins that amount. Keep the group's average where stage one had it.
        nd1 = float((Ou[isg] + Du[isg]).mean())
        shift = (float((solg[:ng] + solg[ng:]).mean()) - nd1) / 2
        Ou[isg], Du[isg] = solg[:ng] - shift, solg[ng:] - shift
        if pack is not None:
            pack["teams"] = [t for t, f in zip(teams, isg) if f]
            pack["rows"] = [[int(gi[ti[q]]) if tg_[q] else -1, int(gi[oi[q]]) if og_[q] else -1, round(float(rr), 5), round(float(ww), 5)]
                            for q, rr, ww in zip(np.flatnonzero(keep), r, w2[keep])]
            pack["kappa"] = [round(float(x), 6) for x in kg]
            pack["ng"] = [int(x) for x in gp[isg]]
            pack["kappa0"], pack["n0"] = float(kappa0), float(n0)
            pack["O0"] = [round(float(x), 5) for x in O0[isg]]
            pack["D0"] = [round(float(x), 5) for x in D0[isg]]
            pack["mu"], pack["b"] = mu, b
            cm = np.array([t in center_on for t in teams]) if center_on else np.ones(n, bool)
            pack["centerO"], pack["centerD"] = float(Ou[cm].mean()), float(Du[cm].mean())
            pack["groupMeanND"] = nd1
            pack["fixedO"] = {t: round(float(x), 4) for t, x, f in zip(teams, Ou, isg) if not f}
            pack["fixedD"] = {t: round(float(x), 4) for t, x, f in zip(teams, Du, isg) if not f}
    cm = np.array([t in center_on for t in teams]) if center_on else np.ones(n, bool)
    O, D = Ou - Ou[cm].mean(), Du - Du[cm].mean()
    return pd.Series(O, teams), pd.Series(D, teams), mu, b


# ---------------------------------------------------------------- production helpers
FG_COEF = (1.0852, -0.8799)      # field goal make probability: expit(b0 + b1 * (distance - 40) / 10), fitted on 2025 and 2026 drives
PUNT_FIT = (122.03, -0.7426)        # expected start (yards to goal) after a punt from `spot`: b0 + b1 * spot
KICK_START = 74.1                # average start (yards to goal) after a kickoff


def _fg_const(dist):
    return expit(FG_COEF[0] + FG_COEF[1] * (np.asarray(dist, float) - 40) / 10)


def points_prior(provider, year, params, group=None, fbs=None):
    """Preseason prior for the points rating: last season's offense and defense, shrunk. None when unavailable."""
    try:
        prev = provider.games(year - 1)
        prev = prev[prev.homePoints.notna() & prev.awayPoints.notna()]
        tg = team_games_from_games(prev, cap=params.od_cap)
        O, D, _, _ = fit_points(tg, int(prev.week.max()), tau=1e9, kappa0=0.4, group=group, center_on=fbs)
        return params.od_share * O, params.od_share * D
    except Exception as e:
        print(f"[warn] no points prior ({type(e).__name__}: {e}); using a flat prior")
        return None


def blended_strength(theta: pd.Series, h: float, played: pd.DataFrame, now_week: int, params, prior=None, group=None, pack=None, fbs=None):
    """Margin rating and points rating in one number, in the margin rating's units.

    Returns dict(R, hfa, O, D, ND): R is the blended strength for every team in `theta`, hfa the home edge to use with it.
    """
    tg = team_games_from_games(played, cap=params.od_cap)
    O, D, mu, b = fit_points(tg, now_week, params.od_tau, params.od_kappa0, params.od_n0, prior, group, 0.0, "pts_c", pack, fbs)
    ND = (O + D).reindex(theta.index).fillna(0.0)
    w = params.od_weight
    R = (1 - w) * theta + w * params.od_scale * ND
    return dict(R=R, hfa=(1 - w) * h + params.od_home, O=O, D=D, ND=ND, mu=mu, b=b)


def special_teams_ratings(provider, year, last_week, games, now_week=None, tau=16.0, kappa0=6.0):
    """Special teams rating per team through `last_week` (points per game against an average opponent), or None."""
    try:
        drives = provider.drives(year, last_week)
        ep = expected_points_curve(provider.api)
        ev, *_ = special_teams_events(drives, ep, _fg_const, PUNT_FIT, KICK_START)
        st = special_teams_by_team_game(ev)
        g = games[games.week <= last_week]
        rating, _ = fit_special(st, g, now_week or last_week, tau=tau, kappa0=kappa0)
        return rating
    except Exception as e:
        print(f"[warn] no special teams rating ({type(e).__name__}: {e})")
        return None


def efficiency_ratings(provider, year, last_week, games, now_week=None, tau=16.0, kappa0=1.0):
    """Offense and defense from play value (PPA) instead of the scoreboard, in points per game; (O, D) or None."""
    try:
        adv = build_team_games(provider.advanced(year, last_week), games)
        O, D, _, _ = fit_off_def(adv[adv.week <= (now_week or last_week)], "o_tppa", now_week or last_week, tau=tau, kappa0=kappa0)
        return O, D
    except Exception as e:
        print(f"[warn] no efficiency rating ({type(e).__name__}: {e})")
        return None


def add_component_ranks(out: pd.DataFrame, provider, year, week, games) -> pd.DataFrame:
    """Special teams and efficiency columns for a week's ranking table (needs net_pts from the pipeline)."""
    out = out.copy()
    st = special_teams_ratings(provider, year, week, games, week)
    if st is not None:
        out["st"] = out.team.map(st)
        out["st_rank"] = out.st.rank(ascending=False, method="min")
    eff = efficiency_ratings(provider, year, week, games, week)
    if eff is not None:
        out["eff"] = out.team.map(eff[0] + eff[1])
        out["eff_rank"] = out.eff.rank(ascending=False, method="min")
        e = out.eff.dropna()
        if len(e):
            slope = float((e * out.net_pts.reindex(e.index)).sum() / (e * e).sum())
            out["luck"] = out.net_pts - slope * out.eff     # positive: the scoreboard is ahead of the play-by-play efficiency
    return out
