#!/usr/bin/env python3
"""Collect odds timelines for sequence autoencoder training."""

import pickle
from pathlib import Path
import pandas as pd

CACHE_DIR = Path("h2h_data/api_cache")
OUT_FILE = CACHE_DIR / "odds_timelines.pkl"


def extract_odds_timelines(cache_dir: Path) -> list[pd.DataFrame]:
    timelines: list[pd.DataFrame] = []
    for fp in cache_dir.glob("*.pkl"):
        try:
            with open(fp, "rb") as f:
                cached = pickle.load(f)
        except Exception as e:  # pragma: no cover - passthrough unexpected errors
            print(f"Error reading {fp}: {e}")
            continue

        if isinstance(cached, dict) and "odds_timeline" in cached:
            timeline = cached["odds_timeline"]
            if isinstance(timeline, pd.DataFrame) and {"timestamp", "price"}.issubset(timeline.columns):
                timelines.append(timeline[["timestamp", "price"]].copy())

        events = (
            cached.get("data")
            if isinstance(cached, dict) and "data" in cached
            else cached
        )
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        timeline = outcome.get("odds_timeline")
                        if isinstance(timeline, pd.DataFrame) and {
                            "price",
                            "timestamp",
                        }.issubset(timeline.columns):
                            timelines.append(timeline[["timestamp", "price"]].copy())
    return timelines


def main() -> None:
    timelines = extract_odds_timelines(CACHE_DIR)
    if not timelines:
        print("No odds timelines found in cache")
        return
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "wb") as f:
        pickle.dump(timelines, f)
    print(f"Saved {len(timelines)} timelines to {OUT_FILE}")


if __name__ == "__main__":
    main()
