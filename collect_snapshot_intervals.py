#!/usr/bin/env python3
"""Periodically fetch historical odds snapshots and cache them."""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

from ml import fetch_historical_h2h_odds

CACHE_DIR = Path("h2h_data/api_cache")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect repeated historical odds snapshots"
    )
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Snapshot interval in minutes",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Total run time in minutes",
    )
    return parser.parse_args(argv)


def utc_now_iso() -> str:
    """Return current UTC time in ISO format with 'Z' suffix."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    interval_sec = args.interval * 60
    end_time = time.time() + args.duration * 60

    while time.time() < end_time:
        date_iso = utc_now_iso()
        events = fetch_historical_h2h_odds(args.sport, date_iso)
        if events:
            ts_str = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            out_file = CACHE_DIR / f"{ts_str}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump({"data": events}, f)
            print(f"Saved {len(events)} events to {out_file}")
        else:
            print("No events found")

        remaining = end_time - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval_sec, remaining))


if __name__ == "__main__":
    main()
