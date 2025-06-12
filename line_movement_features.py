"""Line movement detection utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np


def detect_steam_moves(
    df: pd.DataFrame,
    sportsbook_cols: list[str],
    *,
    window_seconds: int = 600,
    min_books: int = 3,
    move_threshold: float = 0.5,
) -> pd.Series:
    """Return a flag for steam moves within ``window_seconds``."""

    steam_flags = np.zeros(len(df), dtype=int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for i in range(1, len(df)):
        t_now = df.loc[i, "timestamp"]
        t_window = t_now - pd.Timedelta(seconds=window_seconds)
        mask = (df["timestamp"] >= t_window) & (df["timestamp"] < t_now)
        moves = []
        for col in sportsbook_cols:
            prev = df.loc[mask, col].dropna()
            if not prev.empty:
                last_val = prev.iloc[-1]
                change = df.loc[i, col] - last_val
                if abs(change) >= move_threshold:
                    moves.append(np.sign(change))
        if len(moves) >= min_books and (abs(sum(moves)) == len(moves)):
            steam_flags[i] = 1
    return pd.Series(steam_flags, index=df.index, name="steam_move")


def calculate_rlm(
    df: pd.DataFrame,
    consensus_col: str,
    line_col: str,
    public_side_col: str,
) -> pd.Series:
    """Return a reverse line movement flag."""

    line_change = df[line_col] - df[consensus_col]
    rlm = (
        ((df[public_side_col] == 1) & (line_change < 0))
        | ((df[public_side_col] == 0) & (line_change > 0))
    ).astype(int)
    return pd.Series(rlm, index=df.index, name="reverse_line_move")


def add_line_movement_context_features(
    df: pd.DataFrame,
    sportsbook_cols: list[str],
    opening_line_col: str,
    last_move_time_col: str,
) -> pd.DataFrame:
    """Add contextual line movement features."""

    df["net_line_change"] = df[sportsbook_cols].mean(axis=1) - df[opening_line_col]
    df["time_since_last_move"] = (
        df["timestamp"] - df[last_move_time_col]
    ).dt.total_seconds()
    threshold = 0.5
    recent_moves = (df[sportsbook_cols].diff().abs() > threshold).sum(axis=1)
    df["num_books_moved"] = recent_moves
    df["magnitude_recent_moves"] = df[sportsbook_cols].diff().abs().sum(axis=1)
    return df


def compute_odds_volatility(
    df: pd.DataFrame,
    price_cols: list[str] | None = None,
    *,
    window_seconds: int = 3 * 3600,
    min_periods: int = 2,
    count_changes: bool = False,
) -> pd.DataFrame:
    """Return rolling volatility (and optional change counts) for ``price_cols``."""

    if price_cols is None:
        price_cols = [c for c in df.columns if c != "timestamp"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame(index=df.index)

    for col in price_cols:
        roll = (
            df.set_index("timestamp")[col]
            .rolling(f"{window_seconds}s", min_periods=min_periods)
            .std()
            .reset_index(drop=True)
        )
        out[f"volatility_{col}"] = roll

        if count_changes:
            change = (
                df.set_index("timestamp")[col]
                .diff()
                .ne(0)
                .rolling(f"{window_seconds}s", min_periods=1)
                .sum()
                .reset_index(drop=True)
            )
            out[f"changes_{col}"] = change

    return out
