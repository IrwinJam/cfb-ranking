"""CollegeFootballData.com client with an on-disk cache. Reads the key from CFBD_API_KEY or a .env file."""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.collegefootballdata.com"


def _env_file_key():
    f = Path(__file__).resolve().parents[1] / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "CFBD_API_KEY" and v.strip() and "PASTE" not in v:
                return v.strip()
    return None


class CFBD:
    def __init__(self, key=None, cache_dir="data/cache", ttl_hours=6):
        self.key = key or os.environ.get("CFBD_API_KEY") or _env_file_key()
        if not self.key:
            raise RuntimeError("Set CFBD_API_KEY (free key at collegefootballdata.com/key).")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_hours * 3600

    def get(self, path, _ttl_hours=None, **params):
        """_ttl_hours overrides the cache lifetime for this call (finished weeks never change, so they can be kept long)."""
        params = {k: v for k, v in params.items() if v is not None}
        name = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest() + ".json"
        f = self.cache / name
        ttl = self.ttl if _ttl_hours is None else _ttl_hours * 3600
        if f.exists() and time.time() - f.stat().st_mtime < ttl:
            return json.loads(f.read_text(encoding="utf-8"))
        r = requests.get(BASE + path, params=params, headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
        return r.json()


class CFBDProvider:
    def __init__(self, **kw):
        self.api = CFBD(**kw)

    def games(self, year):
        g = pd.DataFrame(self.api.get("/games", year=year, seasonType="regular")).rename(columns={"id": "gameId"})
        return g[["gameId", "week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]].copy()

    def drives(self, year, last_week):
        """FBS drives for weeks 1..last_week. Finished weeks are cached for a month, the newest one for a few hours."""
        rows = []
        for wk in range(1, last_week + 1):
            ttl = 24 * 30 if wk < last_week else None
            rows += self.api.get("/drives", _ttl_hours=ttl, year=year, week=wk, seasonType="regular", classification="fbs")
        return pd.DataFrame(rows)

    def advanced(self, year, last_week):
        """Per game advanced stats (play value, success rate) for FBS teams, weeks 1..last_week."""
        rows = []
        for wk in range(1, last_week + 1):
            ttl = 24 * 30 if wk < last_week else None
            rows += self.api.get("/stats/game/advanced", _ttl_hours=ttl, year=year, week=wk, seasonType="regular", classification="fbs")
        return rows

    def lines(self, year):
        """Mean consensus spread by game (the home team's point margin the market expects)."""
        out = {}
        for r in self.api.get("/lines", _ttl_hours=3, year=year, seasonType="regular"):
            sp = [x["spread"] for x in r.get("lines", []) if x.get("spread") is not None]
            if sp:
                out[(int(r["week"]), r["homeTeam"], r["awayTeam"])] = round(-float(sum(sp) / len(sp)), 1)
        return out

    def teams(self, year):
        t = pd.DataFrame(self.api.get("/teams/fbs", year=year))
        return t[["school", "conference"]]

    def ap(self, year, week):
        data = self.api.get("/rankings", year=year, week=week, seasonType="regular")
        for wk in data:
            for poll in wk["polls"]:
                if poll["poll"] == "AP Top 25":
                    return pd.DataFrame(poll["ranks"])[["rank", "school"]]
        return pd.DataFrame(columns=["rank", "school"])
