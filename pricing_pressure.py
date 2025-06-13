"""Pricing pressure utilities."""


from __future__ import annotations

import pandas as pd
import numpy as np


def price_momentum(df: pd.DataFrame, col: str, window_seconds: int = 3600) -> pd.Series:
    """Return price change over the past ``window_seconds``."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    shifted = df['timestamp'] - pd.Timedelta(seconds=window_seconds)
    df_window = df[['timestamp', col]].copy()
    df_window.columns = ['window_start', 'window_price']
    merged = pd.merge_asof(
        df[['timestamp', col]],
        df_window,
        left_on='timestamp',
        right_on='window_start',
        direction='backward'
    )
    momentum = df[col] - merged['window_price']
    momentum.name = f"momentum_{col}"
    return momentum


def price_acceleration(df: pd.DataFrame, col: str, window_seconds: int = 3600) -> pd.Series:
    """Return the difference in momentum over time."""
    momentum = price_momentum(df, col, window_seconds)
    acceleration = momentum.diff().astype(object)
    if not acceleration.empty:
        acceleration.iloc[0] = None
    acceleration.name = f"acceleration_{col}"
    return acceleration


def cross_book_disparity(df: pd.DataFrame, sharp_col: str, other_cols: list[str]) -> pd.Series:
    """Return disparity between ``sharp_col`` and the mean of ``other_cols``."""
    others_mean = df[other_cols].mean(axis=1)
    disparity = df[sharp_col] - others_mean
    disparity.name = f"disparity_{sharp_col}_vs_others"
    return disparity
