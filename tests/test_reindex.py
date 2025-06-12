import pickle
import warnings
from pathlib import Path

import ml


def _save_dummy_model(path: Path) -> None:
    model = ml.SimpleOddsModel()
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
    assert prob == 100 / (150 + 100)
    assert not w


def test_predict_moneyline_missing(tmp_path):
    path = tmp_path / "model.pkl"
    _save_dummy_model(path)
    features = {"price1": -110}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prob = ml.predict_moneyline_probability(str(path), features)
    assert any("Missing feature columns" in str(msg.message) for msg in w)
    assert abs(prob - (110 / 210)) < 1e-6


class DummyMirrorModel:
    def predict(self, X):
        return X["opening_odds"].values + 10


def _save_dummy_mirror_model(path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(DummyMirrorModel(), f)


def test_extract_advanced_ml_features(tmp_path):
    model_path = tmp_path / "adv.pkl"
    _save_dummy_model(model_path)
    feats = ml.extract_advanced_ml_features(
        str(model_path), price1=150, price2=-170, team1="A", team2="B"
    )
    assert 0 < feats["advanced_ml_prob"] < 1
    assert isinstance(feats["advanced_ml_edge"], float)


def test_extract_market_signals(tmp_path):
    model_path = tmp_path / "mirror.pkl"
    _save_dummy_mirror_model(model_path)
    feats = ml.extract_market_signals(
        str(model_path), price1=150, ticket_percent=60.0
    )
    assert "predicted_mirror_price" in feats
    assert "mirror_score" in feats

