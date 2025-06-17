#!/usr/bin/env python3
"""Collect odds timelines for sequence autoencoder training."""

import argparse
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd


DEFAULT_CACHE_DIR = Path("h2h_data")
DEFAULT_OUT_FILE = DEFAULT_CACHE_DIR / "api_cache" / "odds_timelines.pkl"


def extract_odds_timelines(cache_dir: Path) -> tuple[list[pd.DataFrame], list[str]]:
    """Return timelines and inspected filenames found under ``cache_dir``."""

    def _parse_timestamp(value: Path | str) -> pd.Timestamp | None:
        stem = value.stem if isinstance(value, Path) else str(value)
        try:
            dt = datetime.strptime(stem, "%Y-%m-%dT%H-%M-%SZ")
        except ValueError:
            try:
                dt = datetime.fromisoformat(stem.replace("Z", "+00:00"))
                dt = dt.replace(tzinfo=None)
            except ValueError:
                return None
        return pd.Timestamp(dt)

    timelines: list[pd.DataFrame] = []
    inspected: list[str] = []
    event_rows: dict[str, list[dict]] = {}
    event_files: dict[str, set[str]] = {}

    def _process_snapshot(events, ts: pd.Timestamp | None, label: str) -> None:
        found_local = False
        if isinstance(events, dict):
            events_iter = [events]
        else:
            events_iter = events if isinstance(events, list) else []

        for event in events_iter:
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
                            found_local = True

            if not found_local:
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
                    if price is not None and ts is not None:
                        event_rows.setdefault(event_id, []).append({"timestamp": ts, "price": price})
                        event_files.setdefault(event_id, set()).add(label)

        if not found_local:
            print(f"No odds_timeline in {label}")

    for fp in cache_dir.rglob("*.pkl"):
        inspected.append(fp.name)
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
            else:
                print(f"No odds_timeline in {fp.name}")
            continue

        if isinstance(cached, dict) and "snapshots" in cached:
            for snap in cached["snapshots"]:
                ts = _parse_timestamp(snap.get("timestamp", ""))
                events = snap.get("events") or snap.get("data")
                _process_snapshot(events, ts, fp.name)
        else:
            events = cached.get("data") if isinstance(cached, dict) and "data" in cached else cached
            ts = _parse_timestamp(fp)
            _process_snapshot(events, ts, fp.name)

    # Construct timelines from aggregated snapshot rows
    for event_id, rows in event_rows.items():
        if len(rows) > 0:
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            timelines.append(df[["timestamp", "price"]])

    return timelines, inspected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing cached odds .pkl files",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help="Where to store the aggregated timeline dataset",
    )
    args = parser.parse_args()

    timelines, inspected = extract_odds_timelines(args.cache_dir)
    if not timelines:
        print("No odds timelines found in cache")
        if inspected:
            print("Inspected files: " + ", ".join(sorted(inspected)))
        return
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "wb") as f:
        pickle.dump(timelines, f)
    print(f"Saved {len(timelines)} timelines to {args.out_file}")


if __name__ == "__main__":
    main()
