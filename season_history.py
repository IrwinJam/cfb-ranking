"""Rank the top group for every completed week of a season and write the tables to outputs/<year>/.

    python season_history.py --year 2026
"""
import argparse
from pathlib import Path

import pandas as pd

from aapr.cfbd_client import CFBDProvider
from aapr.params_io import load_params
from aapr.pipeline import run_week
from aapr.schedule import schedule_table

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2026)
ap.add_argument("--prior", choices=["none", "season"], default="season")
ap.add_argument("--params", choices=["fitted", "default"], default="fitted")
a = ap.parse_args()
params = load_params(a.params)
prov = CFBDProvider()

games = prov.games(a.year)
last = int(games[games.homePoints.notna()].week.max())
share_done = games.groupby("week").homePoints.apply(lambda s: s.notna().mean())
if share_done.get(last, 1) < 0.9:
    last -= 1
out = Path("outputs") / str(a.year)
out.mkdir(parents=True, exist_ok=True)

runs, ranks = {}, {}
for w in range(1, last + 1):
    runs[w] = run_week(prov, a.year, w, params, prior=a.prior)
    runs[w]["rankings"].to_csv(out / f"week{w}_aapr.csv", index=False)
    ranks[w] = runs[w]["rankings"].set_index("team")["rank"]

sos_rank = {}
for w in range(1, last + 1):     # schedule strength, with every opponent valued at that week's ratings
    r = runs[w]["rankings"]
    sched = schedule_table(games, runs[w]["theta"], {k: v["theta"] for k, v in runs.items()}, w,
                           r.set_index("team")["rank"], set(r.team), runs[w]["hfa"])
    sched.to_csv(out / f"week{w}_schedule.csv", index=False)
    sos_rank[w] = sched.set_index("team")["sos_all_rank"]

final = runs[last]["rankings"]
pd.DataFrame(ranks).loc[final.team].rename_axis("team").to_csv(out / "rank_history.csv")
pd.DataFrame(sos_rank).loc[final.team].to_csv(out / "schedule_rank_history.csv")
pd.set_option("display.width", 200)
print(final.head(25)[["rank", "team", "conference", "record", "theta", "resume", "score"]].round(2).to_string(index=False))
