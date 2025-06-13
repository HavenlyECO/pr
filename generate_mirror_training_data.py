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
- momentum_price
- acceleration_price
- sharp_disparity
- mirror_target   # closing_odds or line_move
- line_adjustment_rate
- oscillation_frequency
- order_book_imbalance
- market_regime
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from line_movement_features import compute_odds_volatility
from pricing_pressure import (
    price_momentum,
    price_acceleration,
    cross_book_disparity,
)
from liquidity_metrics import (
    line_adjustment_rate,
    oscillation_frequency,
    order_book_imbalance,
)
from market_regime_clustering import derive_regime_features, assign_regime
import joblib
import os

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

                momentum = price_momentum(odds_timeline, "price", window_seconds=3600)
                acceleration = price_acceleration(
                    odds_timeline, "price", window_seconds=3600
                )
                adj_rate_series = line_adjustment_rate(
                    odds_timeline, "price", window_seconds=3600
                )
                osc_freq_series = oscillation_frequency(
                    odds_timeline, "price", threshold=0.1, window_seconds=3600
                )

                regime_feats = derive_regime_features(odds_timeline, "price")
                model_path = "market_regime_model.pkl"
                if os.path.exists(model_path):
                    regime_model = joblib.load(model_path)
                    feature_cols = [col for col in regime_feats.columns]
                    regime_id = assign_regime(regime_feats, regime_model, feature_cols).iloc[0]
                else:
                    regime_id = -1

                if {
                    "sharp_price",
                    "book1_price",
                    "book2_price",
                }.issubset(odds_timeline.columns):
                    disparity = cross_book_disparity(
                        odds_timeline,
                        "sharp_price",
                        ["book1_price", "book2_price"],
                    )
                    disparity_val = disparity.iloc[-1]
                else:
                    disparity_val = 0.0

                momentum_val = momentum.iloc[-1]
                acceleration_val = acceleration.iloc[-1]
                adj_rate = adj_rate_series.iloc[-1]
                osc_freq = osc_freq_series.iloc[-1]
                if {"back_size", "lay_size"}.issubset(odds_timeline.columns):
                    ob_imbalance_series = order_book_imbalance(
                        odds_timeline, "back_size", "lay_size"
                    )
                    ob_imbalance = ob_imbalance_series.iloc[-1]
                else:
                    ob_imbalance = np.nan
            else:
                volatility = np.nan
                momentum_val = acceleration_val = disparity_val = 0.0
                adj_rate = osc_freq = ob_imbalance = np.nan
                regime_id = -1

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
                "momentum_price": momentum_val,
                "acceleration_price": acceleration_val,
                "sharp_disparity": disparity_val,
                "line_adjustment_rate": adj_rate,
                "oscillation_frequency": osc_freq,
                "order_book_imbalance": ob_imbalance,
                "mirror_target": mirror_target,
                "market_regime": regime_id,
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
