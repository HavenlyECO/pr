"""Utilities for clustering market regimes based on line movement."""

from __future__ import annotations

from typing import List

import pandas as pd
import joblib
from sklearn.cluster import KMeans

from line_movement_features import compute_odds_volatility


def total_line_change(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the difference between the final and opening price."""
    change = df[col].iloc[-1] - df[col].iloc[0]
    return pd.Series(change, index=df.index, name=f"total_line_change_{col}")


def largest_move_timing(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the elapsed seconds from start to the largest price move."""
    price_diff = df[col].diff().abs()
    max_idx = price_diff[1:].idxmax()
    elapsed_seconds = (
        df.loc[max_idx, "timestamp"] - df["timestamp"].iloc[0]
    ).total_seconds()
    return pd.Series(elapsed_seconds, index=df.index, name=f"largest_move_timing_{col}")


def derive_regime_features(
    df: pd.DataFrame, price_col: str, *, window_seconds: int = 3 * 3600
) -> pd.DataFrame:
    """Return regime features for ``price_col`` using the final row of ``df``."""
    features = pd.DataFrame(index=df.index)
    df = (
        df.dropna(subset=["timestamp", price_col])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(df) < 2:
        raise ValueError(
            "At least two data points are required to derive regime features"
        )

    # Ensure enough samples exist within the rolling volatility window
    window_start = df["timestamp"].iloc[-1] - pd.Timedelta(seconds=window_seconds)
    count_in_window = (df["timestamp"] >= window_start).sum()
    if count_in_window < 2:
        raise ValueError(
            f"Insufficient data within window: required >=2, found {count_in_window}"
        )

    features[f"total_line_change_{price_col}"] = total_line_change(df, price_col)
    features[f"largest_move_timing_{price_col}"] = largest_move_timing(df, price_col)
    vol = compute_odds_volatility(
        df, price_cols=[price_col], window_seconds=window_seconds
    )
    features[f"volatility_{price_col}"] = vol[f"volatility_{price_col}"]
    if features.iloc[[-1]].isna().any().any():
        raise ValueError("NaN values encountered in derived features")
    event_features = features.iloc[[-1]].reset_index(drop=True)
    return event_features


def train_market_regime_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    n_clusters: int = 3,
    method: str = "kmeans",
    model_path: str = "market_regime_model.pkl",
):
    """Train a clustering model and persist it to ``model_path``."""
    features = df[feature_cols]
    if features.isna().any().any():
        raise ValueError("Training data contains NaNs")
    X = features.values
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    else:
        raise ValueError("Only 'kmeans' is supported in this implementation.")
    model.fit(X)
    joblib.dump(model, model_path)
    return model


def assign_regime(df: pd.DataFrame, model, feature_cols: List[str]) -> pd.Series:
    """Assign a regime cluster to each event in ``df`` using ``model``."""
    if df[feature_cols].isna().any().any():
        raise ValueError("Input data contains NaNs")
    X = df[feature_cols].values
    cluster_ids = model.predict(X)
    return pd.Series(cluster_ids, index=df.index, name="market_regime")
