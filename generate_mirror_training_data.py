#!/usr/bin/env python3
"""
Generate a CSV for market maker mirror model training.

Scans h2h_data/api_cache/*.pkl for events with handle/ticket percentages,
opening odds, and volatility. Outputs a strict dataset for mirror model training.

Required columns:
- opening_odds
- handle_percent
- ticket_percent
- volatility
- mirror_target   # must be defined by your logic (see below)
"""

import pickle
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("h2h_data/api_cache")
OUTPUT_FILE = "mirror_training_data.csv"


def extract_row(event, book, market, home_team, away_team):
    """Extract a row for the mirror model from a single market/book."""
    # Only h2h market supported
    if market.get("key") != "h2h":
        return None

    outcomes = market.get("outcomes", [])
    if len(outcomes) != 2:
        return None

    for outcome in outcomes:
        if outcome.get("name") == home_team:
            team = "home"
        elif outcome.get("name") == away_team:
            team = "away"
        else:
            continue

        opening_odds = outcome.get("opening_price", outcome.get("price"))
        handle_percent = outcome.get("handle_percentage")
        ticket_percent = outcome.get("ticket_percentage")
        volatility = outcome.get("volatility")

        if opening_odds is None or handle_percent is None or ticket_percent is None:
            continue

        team_result = outcome.get("result")
        if team_result is None:
            continue

        mirror_target = 1 if team_result == 1 else 0

        return {
            "opening_odds": opening_odds,
            "handle_percent": handle_percent,
            "ticket_percent": ticket_percent,
            "volatility": volatility,
            "mirror_target": mirror_target,
        }
    return None


def main():
    rows = []
    for cache_file in CACHE_DIR.glob("*.pkl"):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)

        data = cached.get("data") if isinstance(cached, dict) and "data" in cached else cached

        if isinstance(data, dict):
            events = [data]
        elif isinstance(data, list):
            events = data
        else:
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            if not home_team or not away_team:
                continue
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    row = extract_row(event, book, market, home_team, away_team)
                    if row:
                        rows.append(row)

    if not rows:
        print("No eligible rows found. Are handle/ticket percentages populated in your cache?")
        return

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["opening_odds", "handle_percent", "ticket_percent", "mirror_target"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
