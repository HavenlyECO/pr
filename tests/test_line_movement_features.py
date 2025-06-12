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
