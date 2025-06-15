#!/usr/bin/env python3
"""Aggregate daily snapshot odds into event timelines."""

from __future__ import annotations

import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

CACHE_DIR = Path("h2h_data/api_cache")


def _read_snapshot(fp: Path) -> list[dict]:
    """Return event list stored in snapshot ``fp``."""
    with open(fp, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data if isinstance(data, list) else []


def _parse_timestamp(fp: Path) -> pd.Timestamp | None:
    """Parse timestamp from snapshot filename."""
    try:
        dt = datetime.strptime(fp.stem, "%Y-%m-%dT%H-%M-%SZ")
    except ValueError:
        return None
    return pd.Timestamp(dt)


def main() -> None:
    event_rows: dict[str, list[dict]] = {}

    for fp in sorted(CACHE_DIR.glob("*.pkl")):
        ts = _parse_timestamp(fp)
        if ts is None:
            continue
        for event in _read_snapshot(fp):
            if not isinstance(event, dict):
                continue
            event_id = event.get("id")
            if not event_id:
                continue
            price = None
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    if not market.get("outcomes"):
                        continue
                    outcome = market["outcomes"][0]
                    price = outcome.get("price")
                    if price is not None:
                        break
                if price is not None:
                    break
            if price is None:
                continue
            event_rows.setdefault(event_id, []).append({"timestamp": ts, "price": price})

    for event_id, rows in event_rows.items():
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        with open(CACHE_DIR / f"{event_id}.pkl", "wb") as f:
            pickle.dump({"odds_timeline": df}, f)
        print(f"Saved timeline for {event_id}")


if __name__ == "__main__":
    main()
