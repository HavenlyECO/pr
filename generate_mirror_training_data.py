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
    implied_handle,
)
from liquidity_metrics import (
    line_adjustment_rate,
    oscillation_frequency,
    order_book_imbalance,
)
from market_regime_clustering import derive_regime_features, assign_regime
import joblib
import os
from sequence_autoencoder import encode_odds_sequence
from ml import predict_moneyline_probability, market_maker_mirror_score
from market_maker_rl import rl_adjust_price
# Social sentiment and bias features
from social_features import (
    public_bias_score,
    sharp_social_score,
    hype_trend_score,
    lineup_risk_score,
)
# If available:
# from dual_head_nn import predict_dual_head_probability
# from clv_model import predict_clv_probability

CACHE_DIR = Path("h2h_data/api_cache")
OUTPUT_FILE = "mirror_training_data.csv"


def extract_fundamental_features(event):
    """Return a numpy array of fundamental team statistics."""
    return np.array(event["fund_features"], dtype=np.float32)


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
            if odds_timeline is None or len(odds_timeline) <= 1:
                return None

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
            if not os.path.exists(model_path):
                return None
            regime_model = joblib.load(model_path)
            feature_cols = [col for col in regime_feats.columns]
            regime_id = assign_regime(regime_feats, regime_model, feature_cols).iloc[0]

            required_disparity_cols = {"sharp_price", "book1_price", "book2_price"}
            if not required_disparity_cols.issubset(odds_timeline.columns):
                return None
            disparity = cross_book_disparity(
                odds_timeline,
                "sharp_price",
                ["book1_price", "book2_price"],
            )
            disparity_val = disparity.iloc[-1]

            momentum_val = momentum.iloc[-1]
            acceleration_val = acceleration.iloc[-1]
            adj_rate = adj_rate_series.iloc[-1]
            osc_freq = osc_freq_series.iloc[-1]

            if not {"back_size", "lay_size"}.issubset(odds_timeline.columns):
                return None
            ob_imbalance_series = order_book_imbalance(
                odds_timeline, "back_size", "lay_size"
            )
            ob_imbalance = ob_imbalance_series.iloc[-1]

            if opening_odds is None or closing_odds is None:
                continue

            # You may use line_move as the target or closing_odds as the target
            line_move = opening_odds - closing_odds
            # Choose one of the following as your target (mirror_target)
            mirror_target = closing_odds  # Or use line_move if preferred

            row_dict = {
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

            fund_features = extract_fundamental_features(event)
            prices = odds_timeline["price"].to_numpy(dtype=np.float32)
            prices_norm = (prices - prices.mean()) / (
                prices.std() if prices.std() > 1e-5 else 1
            )
            market_array = prices_norm[:, None]
            row_dict["hybrid_fund_features"] = fund_features.tolist()
            row_dict["hybrid_market_array"] = market_array.tolist()

            # Social sentiment and pricing pressure features
            row_dict["home_public_bias"] = public_bias_score(home_team)
            row_dict["away_public_bias"] = public_bias_score(away_team)
            row_dict["home_sharp_social"] = sharp_social_score(home_team)
            row_dict["away_sharp_social"] = sharp_social_score(away_team)
            row_dict["home_hype_trend"] = hype_trend_score(home_team)
            row_dict["away_hype_trend"] = hype_trend_score(away_team)
            row_dict["home_lineup_risk"] = lineup_risk_score(home_team)
            row_dict["away_lineup_risk"] = lineup_risk_score(away_team)
            row_dict["implied_handle"] = implied_handle(
                odds_timeline,
                opening_odds,
                closing_odds,
            )

            # Add multi-scale momentum and volatility features
            for window in [600, 7200, None]:  # 10min, 2hr, full history
                if window is None:
                    mom = odds_timeline["price"].iloc[-1] - odds_timeline["price"].iloc[0]
                    vol = odds_timeline["price"].std()
                    row_dict["momentum_open"] = mom
                    row_dict["volatility_open"] = vol
                else:
                    mom = price_momentum(odds_timeline, "price", window_seconds=window)
                    vol_df = compute_odds_volatility(
                        odds_timeline,
                        price_cols=["price"],
                        window_seconds=window,
                    )
                    row_dict[f"momentum_{window}s"] = float(mom.iloc[-1])
                    row_dict[f"volatility_{window}s"] = float(
                        vol_df["volatility_price"].iloc[-1]
                    )

            if odds_timeline is not None and len(odds_timeline) > 1:
                autoencoder_latent = encode_odds_sequence(
                    odds_timeline, "price", model_path="odds_autoencoder.pt"
                )
                for i, val in enumerate(autoencoder_latent):
                    row_dict[f"autoencoder_feature_{i+1}"] = float(val)

            mm_event = row_dict.copy()
            rl_state = np.array(
                [
                    (
                        pd.to_datetime(event["commence_time"]) - pd.to_datetime(
                            odds_timeline["timestamp"].iloc[-1]
                        )
                    ).total_seconds()
                    / 3600.0,
                    closing_odds / 100.0,
                    volatility,
                    momentum_val,
                ],
                dtype=np.float32,
            )
            fundamental_prob = predict_moneyline_probability(mm_event)
            mirror_score = market_maker_mirror_score(
                "market_maker_mirror.pkl",
                mm_event,
                closing_odds,
            )
            rl_line_adj = rl_adjust_price(rl_state, model_path="market_maker_rl.pt")
            # Optionally:
            # recent_form_prob = predict_dual_head_probability(mm_event)
            # clv_prob = predict_clv_probability(mm_event)
            outcome = event.get("outcome")
            row_dict.update(
                {
                    "fundamental_prob": fundamental_prob,
                    "mirror_score": mirror_score,
                    "rl_line_adjustment": rl_line_adj,
                    # "recent_form_prob": recent_form_prob,
                    # "clv_prob": clv_prob,
                    "home_team_win": 1 if outcome == "home_win" else 0,
                }
            )

            return row_dict
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

    np.savez(
        "hybrid_training_data.npz",
        fund_X=np.stack([np.array(r["hybrid_fund_features"]) for r in rows]),
        market_X=np.stack([np.array(r["hybrid_market_array"]) for r in rows]),
        y=np.array([r["home_team_win"] for r in rows]),
    )


if __name__ == "__main__":
    main()
