#!/usr/bin/env python3
"""Collect odds timelines for sequence autoencoder training."""

import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

CACHE_DIR = Path("h2h_data/api_cache")
OUT_FILE = CACHE_DIR / "odds_timelines.pkl"


def extract_odds_timelines(cache_dir: Path) -> tuple[list[pd.DataFrame], list[str]]:
    """Return timelines and inspected filenames found under ``cache_dir``."""

    def _parse_timestamp(fp: Path) -> pd.Timestamp | None:
        try:
            dt = datetime.fromisoformat(fp.stem)
        except ValueError:
            return None
        return pd.Timestamp(dt)

    timelines: list[pd.DataFrame] = []
    inspected: list[str] = []
    event_rows: dict[str, list[dict]] = {}
    event_files: dict[str, set[str]] = {}
    for fp in cache_dir.glob("*.pkl"):
        inspected.append(fp.name)
        try:
            with open(fp, "rb") as f:
                cached = pickle.load(f)
        except Exception as e:  # pragma: no cover - passthrough unexpected errors
            print(f"Error reading {fp}: {e}")
            continue

        found = False
        if isinstance(cached, dict) and "odds_timeline" in cached:
            timeline = cached["odds_timeline"]
            if isinstance(timeline, pd.DataFrame) and {"timestamp", "price"}.issubset(timeline.columns):
                timelines.append(timeline[["timestamp", "price"]].copy())
                found = True

        events = (
            cached.get("data")
            if isinstance(cached, dict) and "data" in cached
            else cached
        )
        if isinstance(events, dict):
            events = [events]
        if isinstance(events, list):
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
                                found = True

                # Build timeline dynamically from snapshot data
                if not found:
                    event_id = event.get("id")
                    if event_id:
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
                        ts = _parse_timestamp(fp)
                        if price is not None and ts is not None:
                            event_rows.setdefault(event_id, []).append({"timestamp": ts, "price": price})
                            event_files.setdefault(event_id, set()).add(fp.name)

        if not found:
            print(f"No odds_timeline in {fp.name}")

    # Construct timelines from aggregated snapshot rows
    for event_id, rows in event_rows.items():
        if len(rows) > 0:
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            timelines.append(df[["timestamp", "price"]])

    return timelines, inspected


def main() -> None:
    timelines, inspected = extract_odds_timelines(CACHE_DIR)
    if not timelines:
        print("No odds timelines found in cache")
        if inspected:
            print("Inspected files: " + ", ".join(sorted(inspected)))
        return
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "wb") as f:
        pickle.dump(timelines, f)
    print(f"Saved {len(timelines)} timelines to {OUT_FILE}")


if __name__ == "__main__":
    main()
