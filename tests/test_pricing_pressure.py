import pandas as pd

from pricing_pressure import (
    price_momentum,
    price_acceleration,
    cross_book_disparity,
    implied_handle,
)


def test_price_momentum_and_acceleration():
    times = pd.date_range("2021-01-01", periods=3, freq="H")
    df = pd.DataFrame({"timestamp": times, "price": [100, 105, 110]})
    momentum = price_momentum(df, "price", window_seconds=3600)
    acceleration = price_acceleration(df, "price", window_seconds=3600)
    assert list(momentum) == [0, 0, 0]
    assert list(acceleration) == [None, 0, 0]


def test_cross_book_disparity():
    times = pd.date_range("2021-01-01", periods=3, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": times,
            "sharp": [100, 105, 110],
            "book1": [99, 103, 108],
            "book2": [101, 104, 109],
        }
    )
    disparity = cross_book_disparity(df, "sharp", ["book1", "book2"])
    assert list(disparity) == [0.0, 1.5, 1.5]


def test_implied_handle():
    handle = implied_handle(pd.DataFrame(), 200, 150)
    assert abs(handle - 0.25) < 1e-6
    assert implied_handle(pd.DataFrame(), 100, 100) == 0.0
