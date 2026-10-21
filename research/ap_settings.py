"""Which settings of the page (bubble team, strength share, loss penalty) sit closest to the AP poll? Reads rank_review4_tabs.pkl."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr.cfbd_client import CFBD

api = CFBD(ttl_hours=24 * 365)
tabs = pd.read_pickle(Path(__file__).with_name("rank_review4_tabs.pkl"))


def z(s):
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


polls = {}
for t in tabs:
    k = (t["year"], t["week"])
    if k in polls:
        continue
    for entry in api.get("/rankings", year=t["year"], week=t["week"] + 1, seasonType="regular"):
        for poll in entry.get("polls", []):
            if poll.get("poll") == "AP Top 25":
                polls[k] = {x["school"]: int(x["rank"]) for x in poll["ranks"]}


def losses(t):
    g = t["g"]; g = g[g.week <= t["week"]]
    L = pd.concat([g.assign(t=g.homeTeam, l=g.homePoints < g.awayPoints), g.assign(t=g.awayTeam, l=g.awayPoints < g.homePoints)]).groupby("t")["l"].sum()
    return L.reindex(t["teams"]).fillna(0.0)


LOSS = [losses(t) for t in tabs]


def score(i, bubble, mix, pen):
    t = tabs[i]
    return (1 - mix) * z(t["resume"][bubble] - pen * LOSS[i]) + mix * z(t["R"])


def agree(bubble, mix, pen, years):
    r = []
    for i, t in enumerate(tabs):
        ap = polls.get((t["year"], t["week"]))
        if t["year"] not in years or t["week"] < 4 or not ap:
            continue
        s = score(i, bubble, mix, pen)
        inap = [x for x in ap if x in s.index]
        if len(inap) >= 12:
            r.append(spearmanr(s[inap], [-ap[x] for x in inap]).statistic)
    return float(np.mean(r))


res = []
for b, m, p in itertools.product((8, 12, 18, 25, 32, 40), (0.2, 0.3, 0.4, 0.5, 0.6), (0, 1, 2, 3)):
    res.append((agree(b, m, p, {2025, 2026}), agree(b, m, p, {2025}), agree(b, m, p, {2026}), b, m, p))
res.sort(reverse=True)
print("best settings by agreement with the AP poll (both seasons, 2025, 2026):  bubble, strength share, loss penalty")
for r in res[:8]:
    print(f"  {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}   #{r[3]}  {r[4]:.0%}  {r[5]}")
cur = [r for r in res if (r[3], r[4], r[5]) == (12, 0.4, 0)]
print("default-ish (bubble 12, 40%, 0):", [round(x, 3) for x in cur[0][:3]] if cur else None)
b, m, p = res[0][3:]
i = max(j for j, t in enumerate(tabs) if t["year"] == 2026 and t["week"] == 7)
s = score(i, b, m, p).sort_values(ascending=False)
print(f"\nweek 7 of 2026 with bubble #{b}, strength {m:.0%}, loss penalty {p}:", ", ".join(s.index[:12]))
print("AP poll after week 7:", ", ".join(sorted(polls[(2026, 7)], key=polls[(2026, 7)].get)[:12]))
