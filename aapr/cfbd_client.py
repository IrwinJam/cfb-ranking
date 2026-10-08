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

    def get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        name = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest() + ".json"
        f = self.cache / name
        if f.exists() and time.time() - f.stat().st_mtime < self.ttl:
            return json.loads(f.read_text(encoding="utf-8"))
        r = requests.get(BASE + path, params=params, headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
        return r.json()


class CFBDProvider:
    def __init__(self, **kw):
        self.api = CFBD(**kw)

    def games(self, year):
        g = pd.DataFrame(self.api.get("/games", year=year, seasonType="regular"))
        return g[["week", "neutralSite", "homeTeam", "homePoints", "awayTeam", "awayPoints"]].copy()

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
