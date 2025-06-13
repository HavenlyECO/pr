import pandas as pd
import numpy as np


def line_adjustment_rate(df: pd.DataFrame, col: str, *, window_seconds: int = 3600) -> pd.Series:
    """Return the rate of price changes per ``window_seconds`` for ``col``."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    price_changes = df[col].ne(df[col].shift())
    series = price_changes.copy()
    series.index = df["timestamp"]
    roll = series.rolling(f"{window_seconds}s", min_periods=1).sum().reset_index(drop=True)
    rate = roll / (window_seconds / 3600)
    rate.name = f"line_adjustment_rate_{col}"
    return rate


def oscillation_frequency(
    df: pd.DataFrame,
    col: str,
    *,
    threshold: float = 0.1,
    window_seconds: int = 3600,
) -> pd.Series:
    """Return how often price direction alternates within ``window_seconds``."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    price_diff = df[col].diff()
    direction = price_diff.apply(
        lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
    )
    osc = (direction != direction.shift()) & (direction.abs() > 0)
    series = osc.copy()
    series.index = df["timestamp"]
    roll = series.rolling(f"{window_seconds}s", min_periods=1).sum().reset_index(drop=True)
    freq = roll / (window_seconds / 3600)
    freq.name = f"oscillation_frequency_{col}"
    return freq


def order_book_imbalance(df: pd.DataFrame, back_size_col: str, lay_size_col: str) -> pd.Series:
    """Return ``(back_size - lay_size) / (back_size + lay_size)``."""
    back = df[back_size_col]
    lay = df[lay_size_col]
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = (back - lay) / (back + lay)
        imbalance = imbalance.replace([np.inf, -np.inf], np.nan)
    imbalance.name = f"order_book_imbalance_{back_size_col}_{lay_size_col}"
    return imbalance
