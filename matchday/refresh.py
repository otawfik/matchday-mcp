"""Pull the latest football-data.co.uk files into data/ (results for each league and season, and fixtures).

    python -m matchday.refresh
"""
from __future__ import annotations

import urllib.request

from .data import DATA, LEAGUES, SEASONS

BASE = "https://www.football-data.co.uk"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "matchday-mcp (github.com/otawfik)"})
    with urllib.request.urlopen(req, timeout=30) as r:  # follows the site's redirect to the bare domain
        return r.read()


def main() -> None:
    for lg in LEAGUES:
        for s in SEASONS:
            body = fetch(f"{BASE}/mmz4281/{s}/{lg}.csv")
            (DATA / f"{lg}_{s}.csv").write_bytes(body)
            print(f"{lg} {s}: {body.count(b'\n') - 1} matches")
    (DATA / "fixtures.csv").write_bytes(fetch(f"{BASE}/fixtures.csv"))
    print("fixtures.csv updated")


if __name__ == "__main__":
    main()
