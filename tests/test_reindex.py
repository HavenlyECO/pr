import pickle
import warnings
from pathlib import Path

import numpy as np
import ml


class DummyModel:
    def predict_proba(self, X):
        price1 = X["price1"].values[0]
        if price1 > 0:
            implied = 100 / (price1 + 100)
        else:
            implied = abs(price1) / (abs(price1) + 100)
        prob = max(0.0, min(1.0, implied + 0.02))
        return np.array([[1 - prob, prob]])


def _save_dummy_model(path: Path) -> None:
    model = DummyModel()
    cols = ["price1", "foo"]
    with open(path, "wb") as f:
        pickle.dump((model, cols), f)


def test_predict_moneyline_matching(tmp_path):
    path = tmp_path / "model.pkl"
    _save_dummy_model(path)
    features = {"price1": 150, "foo": 1.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        prob = ml.predict_moneyline_probability(str(path), features)
    assert prob == 100 / (150 + 100) + 0.02
    assert not w


def test_predict_moneyline_missing(tmp_path):
    path = tmp_path / "model.pkl"
    _save_dummy_model(path)
    features = {"price1": -110}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prob = ml.predict_moneyline_probability(str(path), features)
    assert any("Missing feature columns" in str(msg.message) for msg in w)
    assert abs(prob - (110 / 210 + 0.02)) < 1e-6




def test_extract_advanced_ml_features(tmp_path):
    model_path = tmp_path / "adv.pkl"
    _save_dummy_model(model_path)
    feats = ml.extract_advanced_ml_features(
        str(model_path), price1=150, price2=-170, team1="A", team2="B"
    )
    assert 0 < feats["advanced_ml_prob"] < 1
    assert isinstance(feats["advanced_ml_edge"], float)


def test_extract_market_signals(tmp_path):
    event = {"opening_price": 150, "volatility": 5.0}
    feats = ml.extract_market_signals(event)
    assert feats["opening_odds"] == 150
    assert feats["volatility"] == 5.0

