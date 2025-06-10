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
