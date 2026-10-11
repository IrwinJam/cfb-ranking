"""Weekly refresh: rerun the season tables, optionally refit the settings, update the README table, optionally push.

    python refresh.py                  # tables and README
    python refresh.py --fit            # also refit the settings (about 25 minutes)
    python refresh.py --push           # also commit and push

The CFBD key goes in .env as CFBD_API_KEY=... (copy .env.example). It is never committed.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
START, END = "<!-- top10:start -->", "<!-- top10:end -->"


def run(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def top10_table(csv):
    r = pd.read_csv(csv).head(10)
    rows = ["| # | Team | Record |", "|---|------|--------|"]
    for _, t in r.iterrows():
        team = t.team if t.conference == "Independent" else f"{t.team} ({t.conference})"
        rows.append(f"| {int(t['rank'])} | {team} | {t.record} |")
    return "\n".join(rows)


def update_readme(year, week):
    f = ROOT / "README.md"
    text = f.read_text(encoding="utf-8")
    text = re.sub(r"## Rankings through week \d+, \d{4}", f"## Rankings through week {week}, {year}", text)
    block = f"{START}\n{top10_table(ROOT / 'outputs' / str(year) / f'week{week}_aapr.csv')}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda m: block, text, flags=re.S)
    else:   # first run: wrap the existing table
        text = re.sub(r"(\| # \| Team \| Record \|\n(?:\|.*\n)+)", lambda m: block + "\n", text, count=1)
    f.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--fit", action="store_true", help="refit the settings to game results")
    ap.add_argument("--push", action="store_true", help="commit and push when done")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from aapr.cfbd_client import CFBDProvider
    try:
        CFBDProvider()
    except RuntimeError as e:
        sys.exit(f"{e}\nPut your key in .env as CFBD_API_KEY=...")

    run(sys.executable, "season_history.py", "--year", str(a.year))
    out = ROOT / "outputs" / str(a.year)
    week = max(int(p.name[4:-9]) for p in out.glob("week*_aapr.csv"))
    if a.fit:
        run(sys.executable, "fit_to_games.py", "--year", str(a.year), "--through", str(week))
    update_readme(a.year, week)
    print(f"Updated through week {week}.")
    if a.push:
        run("git", "add", "-A")
        run("git", "commit", "-m", f"Update through week {week}")
        run("git", "push")


if __name__ == "__main__":
    main()
