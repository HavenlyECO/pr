import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


def train_ensemble_model(dataset_path, model_out="ensemble_model.pkl"):
    """Train a meta-model to blend base model predictions."""
    df = pd.read_csv(dataset_path)
    feature_cols = [
        c for c in df.columns
        if c not in ("home_team_win", "event_id", "date", "target", "game_id")
    ]
    X = df[feature_cols].values
    y = df["home_team_win"].values
    meta = LogisticRegression(class_weight="balanced", max_iter=1000)
    meta.fit(X, y)
    with open(model_out, "wb") as f:
        pickle.dump({
            "model": meta,
            "feature_cols": feature_cols
        }, f)
    print(f"[ensemble_models] Saved ensemble model to {model_out}")


def predict_ensemble_probability(feature_dict, model_path="ensemble_model.pkl"):
    """Predict win probability from base model outputs."""
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    meta = obj["model"]
    feature_cols = obj["feature_cols"]
    X = np.array([[feature_dict.get(col, 0.0) for col in feature_cols]])
    prob = meta.predict_proba(X)[0, 1]
    return float(prob)

