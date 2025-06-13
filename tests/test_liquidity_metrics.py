import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from liquidity_metrics import line_adjustment_rate, oscillation_frequency, order_book_imbalance


def test_line_adjustment_rate_basic():
    times = pd.date_range("2021-01-01", periods=5, freq="30min")
    df = pd.DataFrame({"timestamp": times, "price": [100, 100, 101, 99, 99]})
    rate = line_adjustment_rate(df, "price", window_seconds=3600)
    assert list(rate) == [1, 1, 1, 2, 1]


def test_oscillation_frequency_basic():
    times = pd.date_range("2021-01-01", periods=5, freq="h")
    df = pd.DataFrame({"timestamp": times, "price": [100, 102, 101, 103, 102]})
    freq = oscillation_frequency(df, "price", threshold=0.1, window_seconds=3600)
    assert list(freq) == [0, 1, 1, 1, 1]


def test_order_book_imbalance():
    df = pd.DataFrame({"back": [200, 100], "lay": [100, 100]})
    imbalance = order_book_imbalance(df, "back", "lay")
    assert np.isclose(imbalance.iloc[0], 1/3)
    assert imbalance.iloc[1] == 0
