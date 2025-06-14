#!/usr/bin/env python3
# Stores daily snapshot odds only; timelines require ``fetch_odds_timelines.py``.
"""Fetch historical odds from The Odds API and save them to the cache."""

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from ml import fetch_historical_h2h_odds, to_fixed_utc

CACHE_DIR = Path("h2h_data/api_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache historical odds")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD (inclusive)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start_date)
    end = datetime.fromisoformat(args.end_date) if args.end_date else start

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    day = start
    while day <= end:
        date_iso = to_fixed_utc(day)
        events = fetch_historical_h2h_odds(args.sport, date_iso)
        out_file = CACHE_DIR / f"{day.date()}.pkl"
        if events:
            with open(out_file, "wb") as f:
                pickle.dump({"data": events}, f)
            print(f"Saved {len(events)} events to {out_file}")
        else:
            print(f"No events found for {day.date()}")
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
