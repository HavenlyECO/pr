#!/usr/bin/env python3
"""Create a Retrosheet training CSV with real odds.

The script fetches ``GLYYYY.TXT`` gamelogs for seasons 2018–2025 unless they
already exist under ``retrosheet_data``. It then pulls historical moneyline
prices from The Odds API (when ``THE_ODDS_API_KEY`` is configured) and merges
them with the Retrosheet logs. Rolling team metrics along with manager and
umpire info are included and the final dataset is written to
``retrosheet_training_data.csv``.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("retrosheet_data")
ODDS_CACHE_DIR = DATA_DIR / "odds_cache"
API_KEY = os.getenv("THE_ODDS_API_KEY")
OUTPUT_FILE = "retrosheet_training_data.csv"
YEARS = range(2018, 2026)  # inclusive
BASE = "https://www.retrosheet.org/gamelogs"


def find_gamelog_file(year: int) -> Path | None:
    """Return gamelog path for ``year`` ignoring case, if present."""
    pattern = f"GL{year}".lower()
    for p in DATA_DIR.glob("*.txt"):
        if p.stem.lower() == pattern:
            return p
    return None


def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def fetch_odds_for_date(date: str) -> list:
    """Return odds JSON for ``date`` using cache when possible."""
    ODDS_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = ODDS_CACHE_DIR / f"{date}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    if not API_KEY:
        print(f"No API key, skipping odds for {date}")
        return []
    url = (
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history"
        f"/?apiKey={API_KEY}&regions=us&markets=h2h&date={date}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.ok:
            cache_file.write_text(resp.text)
            return resp.json()
        print(f"Failed to fetch odds for {date}: {resp.status_code}")
    except Exception as exc:
        print(f"Error fetching odds for {date}: {exc}")
    return []


def build_odds_dataframe(dates: list[str]) -> pd.DataFrame:
    """Fetch odds for ``dates`` and return as a dataframe."""
    records = []
    for d in sorted(set(dates)):
        for event in fetch_odds_for_date(d):
            home = event.get("home_team")
            away = event.get("away_team")
            if not home or not away:
                continue
            price1 = price2 = None
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home:
                            price1 = outcome.get("price")
                        elif outcome.get("name") == away:
                            price2 = outcome.get("price")
                    if price1 is not None and price2 is not None:
                        break
                if price1 is not None and price2 is not None:
                    break
            if price1 is None or price2 is None:
                continue
            records.append({
                "date": event.get("commence_time", "")[:10],
                "home_team": home,
                "visiting_team": away,
                "pregame_price": price1,
                "away_price": price2,
                "implied_prob": american_odds_to_prob(price1),
            })
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=["date", "home_team", "visiting_team", "pregame_price", "away_price", "implied_prob"])


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    frames = []
    for year in YEARS:
        txt = find_gamelog_file(year)
        if not txt:
            zip_path = DATA_DIR / f"gl{year}.zip"
            if not zip_path.exists():
                url = f"{BASE}/gl{year}.zip"
                print(f"Downloading {url}…")
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.ok:
                        zip_path.write_bytes(resp.content)
                    else:
                        print(f"Failed to fetch {url}")
                        continue
                except Exception as exc:  # pragma: no cover
                    print(f"Error downloading {year}: {exc}")
                    continue
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    name = next(
                        (n for n in zf.namelist() if n.upper().startswith("GL") and n.upper().endswith(".TXT")),
                        None,
                    )
                    if name:
                        txt = DATA_DIR / f"GL{year}.TXT"
                        txt.write_bytes(zf.read(name))
                    else:
                        print(f"No TXT found in {zip_path}")
                        continue
            except Exception as exc:  # pragma: no cover
                print(f"Error extracting {zip_path}: {exc}")
                continue
        else:
            print(f"Using existing {txt.name}")

        if not txt.exists():
            print(f"Missing gamelog for {year}")
            continue

        df = pd.read_csv(txt, header=None)
        df = df.rename(columns={
            0: "date",
            1: "game_number",
            2: "day_of_week",
            3: "visiting_team",
            4: "visiting_league",
            5: "visiting_game_num",
            6: "home_team",
            7: "home_league",
            8: "home_game_num",
            9: "visiting_score",
            10: "home_score",
            11: "outs",
            12: "day_night",
            16: "park_id",
            17: "attendance",
            80: "visiting_manager",
            82: "home_manager",
            90: "home_plate_umpire",
        })
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df["year"] = year
        df = df.dropna(subset=["visiting_score", "home_score"])
        df["home_team_win"] = (df["home_score"] > df["visiting_score"]).astype(int)
        frames.append(df)

    if not frames:
        raise SystemExit("No Retrosheet data processed")

    df = pd.concat(frames)

    # rolling team statistics
    games = []
    home = df[["date", "home_team", "visiting_team", "home_score", "visiting_score", "home_team_win"]].copy()
    home.columns = ["date", "team", "opp", "runs_scored", "runs_allowed", "win"]
    home["is_home"] = 1
    games.append(home)
    away = df[["date", "visiting_team", "home_team", "visiting_score", "home_score", "home_team_win"]].copy()
    away.columns = ["date", "team", "opp", "runs_scored", "runs_allowed", "home_win"]
    away["win"] = 1 - away.pop("home_win")
    away["is_home"] = 0
    games.append(away)

    long = pd.concat(games).sort_values("date")
    grp = long.groupby("team")
    for n in (5, 10):
        long[f"rolling_win_pct_{n}"] = grp["win"].apply(lambda s: s.shift().rolling(n, 1).mean())
        long[f"rolling_runs_scored_{n}"] = grp["runs_scored"].apply(lambda s: s.shift().rolling(n, 1).mean())
        long[f"rolling_runs_allowed_{n}"] = grp["runs_allowed"].apply(lambda s: s.shift().rolling(n, 1).mean())
        long[f"rolling_run_diff_{n}"] = long[f"rolling_runs_scored_{n}"] - long[f"rolling_runs_allowed_{n}"]

    home_feats = long[long["is_home"] == 1].drop(columns=["opp", "runs_scored", "runs_allowed", "win", "is_home"]).rename(columns={"team": "home_team"})
    away_feats = long[long["is_home"] == 0].drop(columns=["opp", "runs_scored", "runs_allowed", "win", "is_home"]).rename(columns={"team": "visiting_team"})

    df = df.merge(home_feats, on=["date", "home_team"], how="left")
    df = df.merge(away_feats, on=["date", "visiting_team"], how="left", suffixes=("_home", "_away"))

    defaults = {
        "rolling_win_pct_5": 0.5,
        "rolling_win_pct_10": 0.5,
        "rolling_run_diff_5": 0.0,
        "rolling_run_diff_10": 0.0,
        "rolling_runs_scored_5": 4.5,
        "rolling_runs_allowed_5": 4.5,
    }
    for col, val in defaults.items():
        df[f"home_{col}"] = df[f"home_{col}"].fillna(val)
        df[f"visiting_{col}"] = df[f"visiting_{col}"].fillna(val)

    odds_df = build_odds_dataframe(df['date'].dt.strftime('%Y-%m-%d').unique().tolist())
    df["date_str"] = df["date"].dt.strftime('%Y-%m-%d')
    df = df.merge(
        odds_df,
        left_on=["date_str", "home_team", "visiting_team"],
        right_on=["date", "home_team", "visiting_team"],
        how="left",
    )
    df = df.drop(columns=["date", "date_str"], errors="ignore")
    df["pregame_line"] = df["pregame_price"]
    df["implied_prob"] = df["pregame_price"].apply(lambda x: american_odds_to_prob(x) if pd.notna(x) else None)
    df["away_team"] = df["visiting_team"]
    df["team1"] = df["home_team"]
    df["team2"] = df["away_team"]
    df["pregame_win_pct_diff"] = df["home_rolling_win_pct_10"] - df["visiting_rolling_win_pct_10"]
    df["pregame_run_diff"] = df["home_rolling_run_diff_10"] - df["visiting_rolling_run_diff_10"]
    df["pregame_home_adv"] = 1.0
    df["pregame_day_night"] = (df["day_night"] == "N").astype(float)
    df["live_inning_5_diff"] = None
    df["live_inning_7_diff"] = None

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} games to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
