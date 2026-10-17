"""Pull the extra CollegeFootballData tables the component model needs, once, into the on-disk cache.

  python research/fetch_components.py            # 2025 full season + 2026 so far
Each call is cached for a year, so reruns cost nothing. The free tier allows 1,000 calls a month.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aapr.cfbd_client import CFBD

api = CFBD(ttl_hours=24 * 365)


def weeks_for(year):
    return range(1, 17) if year < 2026 else range(1, 8)


calls = 0
for year in (2025, 2026):
    for wk in weeks_for(year):
        for path, kw in (("/stats/game/advanced", dict(classification="fbs")), ("/drives", dict(classification="fbs"))):
            d = api.get(path, year=year, week=wk, seasonType="regular", **kw)
            calls += 1
            print(year, wk, path, len(d), flush=True)
    for path, kw in (("/lines", dict(seasonType="regular")), ("/talent", {})):
        d = api.get(path, year=year, **kw)
        print(year, path, len(d), flush=True)
api.get("/ppa/predicted", down=1, distance=10)
print("done")
