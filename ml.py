import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Functions only; no code at global scope except imports and definitions.

def train_mvp_model(csv_path: str, model_path: str) -> None:
    """Train a logistic regression model and persist it to ``model_path``."""
    df = pd.read_csv(csv_path)
    required_cols = ["price1", "price2", "home_team_win"]

    # Strict: fail if columns missing
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[["price1", "price2"]]
    y = df["home_team_win"]

    model = LogisticRegression()
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def predict_mvp(model_path: str, price1: float, price2: float) -> float:
    """Return the predicted home win probability for the given prices."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X_pred = pd.DataFrame([{"price1": price1, "price2": price2}])
    return model.predict_proba(X_pred)[:, 1][0]

# Usage example (commented out; use in your CLI or pipeline)
# train_mvp_model("retrosheet_training_data.csv", "mvp_model.pkl")
# prob = predict_mvp("mvp_model.pkl", -120, 110)
# print("Win probability:", prob)
