from dataclasses import dataclass

POWER_FOUR = ("ACC", "Big Ten", "Big 12", "SEC")
INDEPENDENT = "Independent"
INDEPENDENT_TEAMS = ("Notre Dame",)
TOP_GROUP = POWER_FOUR + (INDEPENDENT,)


@dataclass
class Params:
    k: float = 4.0                  # margin compression: k * sign * ln(1 + |margin|)
    tau: float = 6.0                # recency decay, in weeks
    kappa0: float = 3.0             # ridge strength toward the prior at the start of the season
    n0: float = 4.0                 # games played at which that strength halves
    sigma_g: float = 11.0           # game noise, in compressed units
    prior_shrink: float = 0.6       # share of last season's rating kept in the prior
    beta_early: float = 0.30        # weight on rating versus resume, early in the season
    beta_late: float = 0.50
    beta_ramp_weeks: int = 12
    sos_weight: float = 0.0        # share of the score from the schedule term: 0, since the resume and the strength rating already price the schedule (docs/research.md)
    loss_penalty: float = 0.0       # resume points off for every loss; tried at 2 and reverted, see docs/research.md
    group_game_weight: float = 1.5  # games between two top-group teams
    hfa_scale: float = 1.0          # scale on the estimated home-field edge
    upset_bonus: float = 0.0        # extra weight when the lower-rated team wins
    nonp4_win_weight: float = 0.0   # weight of a top-group win over an outside team
    # points rating (offense and defense), blended into the margin rating; see docs/research.md
    od_weight: float = 0.8          # share of the blended strength that comes from the points rating
    od_scale: float = 0.375         # margin-rating units per point of net offense plus defense
    od_share: float = 0.8           # share of last season's offense and defense carried into the prior
    od_kappa0: float = 6.0          # ridge strength of the points rating at the start of the season
    od_n0: float = 4.0
    od_tau: float = 16.0            # recency decay in weeks
    od_cap: float = 56.0            # points scored are capped here
    od_home: float = -0.25          # home edge added to the blended rating
    sigma_blend: float = 5.75       # game noise of the blended rating
