#!/usr/bin/env python3
"""
Generate a CSV for market maker mirror model training.

Scans h2h_data/api_cache/*.pkl for events with opening/closing odds and volatility.
Outputs a dataset for mirror model training.

Columns:
- opening_odds
- closing_odds
- line_move (opening_odds - closing_odds)
- volatility
- mirror_target   # closing_odds or line_move
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from line_movement_features import compute_odds_volatility

CACHE_DIR = Path("h2h_data/api_cache")
OUTPUT_FILE = "mirror_training_data.csv"


def extract_row(event, book, market, home_team, away_team):
    """Extract a row for the mirror model from a single market/book."""
    if market.get("key") != "h2h":
        return None

    outcomes = market.get("outcomes", [])
    if len(outcomes) != 2:
        return None

    for outcome in outcomes:
        if outcome.get("name") == home_team or outcome.get("name") == away_team:
            opening_odds = outcome.get("opening_price", outcome.get("price"))
            closing_odds = outcome.get("closing_price")

            odds_timeline = outcome.get("odds_timeline")
            if odds_timeline is not None and len(odds_timeline) > 1:
                vol_df = compute_odds_volatility(
                    odds_timeline,
                    price_cols=["price"],
                    window_seconds=3 * 3600,
                )
                volatility = vol_df["volatility_price"].iloc[-1]
            else:
                volatility = np.nan

            if opening_odds is None or closing_odds is None:
                continue

            # You may use line_move as the target or closing_odds as the target
            line_move = opening_odds - closing_odds
            # Choose one of the following as your target (mirror_target)
            mirror_target = closing_odds  # Or use line_move if preferred

            return {
                "opening_odds": opening_odds,
                "closing_odds": closing_odds,
                "line_move": line_move,
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
        print("No eligible rows found. Are opening/closing odds populated in your cache?")
        return

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["opening_odds", "closing_odds", "volatility", "mirror_target"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
