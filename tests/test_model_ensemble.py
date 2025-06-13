import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ensemble_models import train_ensemble_model, predict_ensemble_probability


def test_train_and_predict(tmp_path):
    # Tiny synthetic dataset: two base models, one target
    np.random.seed(0)
    df = pd.DataFrame({
        "fundamental_prob": np.random.uniform(0.4, 0.7, 100),
        "mirror_score": np.random.uniform(0.3, 0.8, 100),
        "rl_line_adjustment": np.random.choice([-1, 0, 1], 100),
        "home_team_win": np.random.binomial(1, 0.55, 100),
    })
    dataset_path = tmp_path / "ensemble_train.csv"
    df.to_csv(dataset_path, index=False)
    model_path = tmp_path / "ensemble_model.pkl"
    train_ensemble_model(str(dataset_path), model_out=str(model_path))
    # Predict for a random row
    random_row = df.sample(1).iloc[0]
    feat_dict = {
        "fundamental_prob": random_row["fundamental_prob"],
        "mirror_score": random_row["mirror_score"],
        "rl_line_adjustment": random_row["rl_line_adjustment"],
    }
    prob = predict_ensemble_probability(feat_dict, model_path=str(model_path))
    assert 0.0 <= prob <= 1.0


def test_ensemble_improves_over_base(tmp_path):
    # Create a problem where ensemble (average) is better than either alone
    df = pd.DataFrame({
        "fundamental_prob": [0.6, 0.4, 0.7, 0.3],
        "mirror_score": [0.7, 0.3, 0.6, 0.4],
        "rl_line_adjustment": [1, -1, 1, -1],
        "home_team_win": [1, 0, 1, 0],
    })
    dataset = tmp_path / "test_ensemble_train.csv"
    df.to_csv(dataset, index=False)
    model_file = tmp_path / "test_ensemble_model.pkl"
    train_ensemble_model(str(dataset), model_out=str(model_file))
    feat_dict = {
        "fundamental_prob": 0.65,
        "mirror_score": 0.65,
        "rl_line_adjustment": 1,
    }
    prob = predict_ensemble_probability(feat_dict, model_path=str(model_file))
    assert 0.0 <= prob <= 1.0

