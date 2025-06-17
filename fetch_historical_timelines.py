#!/usr/bin/env python3
"""Collect long-range odds timelines via the historical endpoint."""

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

import ml

ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if load_dotenv and DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

API_KEY = os.getenv("THE_ODDS_API_KEY")
CACHE_DIR = Path("h2h_data") / "api_cache"
DEFAULT_OUT_FILE = CACHE_DIR / "snapshot_data.pkl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical odds timelines across a date range"
    )
    parser.add_argument("--start-date", required=True, help="First date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Last date YYYY-MM-DD")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument(
        "--interval",
        type=int,
        help="Minute interval for repeated snapshots within the range",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help="Aggregated output file for all timelines",
    )
    return parser.parse_args(argv)


def _fetch_odds_history(sport: str, event_id: str) -> list | dict:
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


def daterange(start: datetime, end: datetime, *, minutes: int | None = None):
    step = timedelta(minutes=minutes) if minutes else timedelta(days=1)
    current = start
    while current <= end:
        yield current
        current += step


def iso_timestamp(ts: datetime, *, minutes: int | None = None) -> str:
    if minutes:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ml.to_fixed_utc(ts)


def merge_timelines(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([existing, new], ignore_index=True)
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = datetime.fromisoformat(args.start_date)
    end = datetime.fromisoformat(args.end_date)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Collecting odds timelines for",
        f" {args.sport} from {args.start_date} to {args.end_date}"
    )
    print(f"Saving data to {CACHE_DIR.resolve()}")

    aggregated: dict[str, pd.DataFrame] = {}

    for ts in daterange(start, end, minutes=args.interval):
        date_iso = iso_timestamp(ts, minutes=args.interval)
        print(f"Fetching snapshot for {date_iso}")
        events = ml.fetch_historical_h2h_odds(args.sport, date_iso)
        print(f"Found {len(events)} events")
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id")
            if not event_id:
                continue
            print(f"Fetching history for {event_id}")
            try:
                hist = _fetch_odds_history(args.sport, event_id)
            except Exception as exc:  # pragma: no cover - network error handling
                print(f"Error fetching history for {event_id}: {exc}")
                continue
            df = pd.DataFrame(hist)
            cache_path = CACHE_DIR / f"{event_id}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and isinstance(data.get("odds_timeline"), pd.DataFrame):
                    df = merge_timelines(data["odds_timeline"], df)
                    print(f"Merging with existing {cache_path}")

            data = {"odds_timeline": df}
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved timeline for {event_id} to {cache_path.resolve()}")
            if event_id in aggregated:
                aggregated[event_id] = merge_timelines(aggregated[event_id], df)
            else:
                aggregated[event_id] = df

    if aggregated:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "wb") as f:
            pickle.dump(list(aggregated.values()), f)
        print(f"Saved {len(aggregated)} timelines to {args.out_file}")


if __name__ == "__main__":
    main()
