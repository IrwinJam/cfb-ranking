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
    sos_weight: float = 0.15        # hand-set, not fitted
    group_game_weight: float = 1.5  # games between two top-group teams
    hfa_scale: float = 1.0          # scale on the estimated home-field edge
    upset_bonus: float = 0.0        # extra weight when the lower-rated team wins
    nonp4_win_weight: float = 0.0   # weight of a top-group win over an outside team
