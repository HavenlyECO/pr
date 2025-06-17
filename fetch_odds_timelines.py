#!/usr/bin/env python3
"""Download historical odds timelines for each event."""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if load_dotenv and DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

API_KEY = os.getenv("THE_ODDS_API_KEY")
CACHE_DIR = Path("h2h_data") / "api_cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch odds timelines")
    parser.add_argument("--start-date", required=True, help="First date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Last date YYYY-MM-DD")
    parser.add_argument(
        "--sport",
        default="baseball_mlb",
        help="Sport key used by The Odds API",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        help="Optional aggregated output file for all timelines",
    )
    return parser.parse_args(argv)


def to_fixed_utc(date_obj: datetime) -> str:
    """Return ISO string for ``date_obj`` fixed at 12:00 UTC."""
    return date_obj.strftime("%Y-%m-%dT12:00:00Z")


def fetch_historical_odds(sport: str, date_iso: str) -> list:
    """Return odds data for ``sport`` on ``date_iso``."""
    if not API_KEY:
        raise RuntimeError("THE_ODDS_API_KEY environment variable is not set")
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": date_iso,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_odds_history(sport: str, event_id: str) -> list | dict:
    """Return odds history for ``event_id``."""
    if not API_KEY:
        raise RuntimeError("THE_ODDS_API_KEY environment variable is not set")
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds-history"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def daterange(start: datetime, end: datetime):
    day = timedelta(days=1)
    current = start
    while current <= end:
        yield current
        current += day


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = datetime.fromisoformat(args.start_date)
    end = datetime.fromisoformat(args.end_date)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Individual event files will be saved to {CACHE_DIR}")
    if args.out_file is not None:
        print(f"Aggregated timelines will be written to {args.out_file}")

    aggregated: list[pd.DataFrame] = []

    for d in daterange(start, end):
        date_iso = to_fixed_utc(d)
        try:
            events = fetch_historical_odds(args.sport, date_iso)
        except Exception as exc:  # pragma: no cover - network error handling
            print(f"Error fetching events for {d.date()}: {exc}")
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id")
            if not event_id:
                continue
            cache_path = CACHE_DIR / f"{event_id}.pkl"
            if cache_path.exists():
                print(f"Using existing {cache_path}")
                continue
            try:
                hist = fetch_odds_history(args.sport, event_id)
            except Exception as exc:  # pragma: no cover - network error handling
                print(f"Error fetching history for {event_id}: {exc}")
                continue
            df = pd.DataFrame(hist)
            data = {"odds_timeline": df}
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved timeline for {event_id} to {cache_path}")
            if args.out_file is not None:
                aggregated.append(df)

    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "wb") as f:
            pickle.dump(aggregated, f)
        print(f"Saved {len(aggregated)} timelines to {args.out_file}")


if __name__ == "__main__":
    main()
