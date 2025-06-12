import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from line_movement_features import (
    detect_steam_moves,
    calculate_rlm,
    add_line_movement_context_features,
    compute_odds_volatility,
)


def test_detect_steam_moves_basic():
    now = pd.Timestamp.utcnow()
    data = {
        "timestamp": [now + timedelta(seconds=i * 300) for i in range(4)],
        "book1": [100, 101, 102, 102],
        "book2": [100, 101, 102, 102],
        "book3": [100, 101, 102, 102],
    }
    df = pd.DataFrame(data)
    flags = detect_steam_moves(df, ["book1", "book2", "book3"], window_seconds=600)
    assert flags.iloc[2] == 1
    assert flags.sum() == 2


def test_calculate_rlm():
    df = pd.DataFrame(
        {
            "consensus": [100, 100, 100],
            "current": [99, 101, 100],
            "public": [1, 0, 1],
        }
    )
    rlm = calculate_rlm(df, "consensus", "current", "public")
    assert list(rlm) == [1, 1, 0]


def test_add_line_movement_context_features():
    now = pd.Timestamp.utcnow()
    df = pd.DataFrame(
        {
            "timestamp": [now, now + timedelta(seconds=60)],
            "open": [100, 100],
            "last_move": [now - timedelta(seconds=30), now],
            "book1": [100, 100.6],
            "book2": [100, 100.4],
        }
    )
    result = add_line_movement_context_features(
        df.copy(), ["book1", "book2"], "open", "last_move"
    )
    assert "net_line_change" in result
    assert "num_books_moved" in result
    assert result.loc[1, "num_books_moved"] == 1


def test_compute_odds_volatility_basic():
    now = pd.Timestamp.utcnow()
    df = pd.DataFrame(
        {
            "timestamp": [now, now + timedelta(seconds=60), now + timedelta(seconds=120)],
            "p1": [100, 110, 120],
            "p2": [200, 200, 200],
        }
    )
    result = compute_odds_volatility(
        df, ["p1", "p2"], window_seconds=120, min_periods=2
    )
    assert "volatility_p1" in result
    assert "volatility_p2" in result
    assert np.isnan(result.loc[0, "volatility_p1"])
    assert abs(result.loc[1, "volatility_p1"] - 7.0710678118654755) < 1e-6
    assert abs(result.loc[2, "volatility_p1"] - 7.0710678118654755) < 1e-6
    assert result.loc[1, "volatility_p2"] == 0.0
    assert result.loc[2, "volatility_p2"] == 0.0


def test_compute_odds_volatility_count_changes():
    now = pd.Timestamp.utcnow()
    df = pd.DataFrame(
        {
            "timestamp": [now, now + timedelta(seconds=60), now + timedelta(seconds=120)],
            "p1": [100, 110, 120],
        }
    )
    result = compute_odds_volatility(
        df, ["p1"], window_seconds=120, count_changes=True
    )
    assert "changes_p1" in result
    assert list(result["changes_p1"]) == [1.0, 2.0, 2.0]
