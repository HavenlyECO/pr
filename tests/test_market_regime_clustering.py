import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from market_regime_clustering import (
    total_line_change,
    largest_move_timing,
    derive_regime_features,
    train_market_regime_model,
    assign_regime,
)


def test_total_line_change():
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=5, freq="H"), "price": [100, 101, 102, 103, 105]})
    tlc = total_line_change(df, "price")
    assert tlc.iloc[0] == 5


def test_largest_move_timing():
    times = pd.date_range("2023-01-01", periods=5, freq="H")
    prices = [100, 101, 120, 121, 122]
    df = pd.DataFrame({"timestamp": times, "price": prices})
    lmt = largest_move_timing(df, "price")
    assert lmt.iloc[0] == (times[2] - times[0]).total_seconds()


def test_regime_clustering_and_assignment():
    data = []
    for offset in [0, 10, 20]:
        price = [100 + offset] * 5
        if offset == 10:
            price = [110, 120, 120, 120, 120]
        if offset == 20:
            price = [120, 120, 120, 120, 130]
        times = pd.date_range("2023-01-01", periods=5, freq="H")
        df = pd.DataFrame({"timestamp": times, "price": price})
        feats = derive_regime_features(df, "price")
        data.append(feats)
    cluster_df = pd.concat(data, ignore_index=True)
    feature_cols = list(cluster_df.columns)
    model = train_market_regime_model(cluster_df, feature_cols, n_clusters=3, method="kmeans", model_path="test_market_regime_model.pkl")
    labels = assign_regime(cluster_df, model, feature_cols)
    assert len(labels.unique()) == 3
    os.remove("test_market_regime_model.pkl")
