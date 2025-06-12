import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import pickle
import warnings

# Access model path constants from main without creating a circular import
from typing import Optional

# Functions only; no code at global scope except imports and definitions.

def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to an implied win probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_odds_to_payout(odds: float) -> float:
    """Return the profit on a $1 bet for the given American odds."""
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)

def train_mvp_model(csv_path: str, model_path: Optional[str] = None) -> None:
    """Train a logistic regression model and persist it."""
    from main import H2H_MODEL_PATH
    model_path = model_path or str(H2H_MODEL_PATH)
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

def predict_mvp(
    model_path: Optional[str] = None, price1: float | int = 0, price2: float | int = 0
) -> float:
    """Return the predicted home win probability for the given prices."""
    from main import H2H_MODEL_PATH
    model_path = model_path or str(H2H_MODEL_PATH)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X_pred = pd.DataFrame([{"price1": price1, "price2": price2}])
    return model.predict_proba(X_pred)[:, 1][0]


def predict_moneyline_probability(
    model_path: Optional[str] = None, features: dict | None = None
) -> float:
    """Predict win probability using a trained moneyline classifier."""
    from main import MONEYLINE_MODEL_PATH
    model_path = model_path or str(MONEYLINE_MODEL_PATH)
    features = features or {}
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)
    if isinstance(model_info, tuple):
        model, cols = model_info
    else:
        model = model_info
        cols = None
    df = pd.DataFrame([features])
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"Missing feature columns: {', '.join(missing)}",
                RuntimeWarning,
            )
        df = df.reindex(cols, axis=1, fill_value=0)
    return float(model.predict_proba(df)[0][1])


def extract_advanced_ml_features(
    model_path: Optional[str] = None,
    *,
    price1: float,
    price2: float,
    team1: str | None = None,
    team2: str | None = None,
) -> dict:
    """Return basic advanced metrics for given prices."""
    from main import MONEYLINE_MODEL_PATH
    model_path = model_path or str(MONEYLINE_MODEL_PATH)
    prob = predict_moneyline_probability(
        model_path,
        {"price1": price1, "price2": price2},
    )
    implied = american_odds_to_prob(price1)
    edge = prob - implied
    ev = edge * american_odds_to_payout(price1)
    return {
        "advanced_ml_prob": prob,
        "advanced_ml_edge": edge,
        "advanced_ml_ev": ev,
        "market_efficiency": implied,
        "sharp_action": edge,
        "ml_confidence": prob,
        "lineup_strength": 0.5,
    }


def extract_market_signals(
    model_path: Optional[str] = None,
    *,
    price1: float,
    handle_percent: float | None,
    ticket_percent: float | None,
) -> dict:
    """Return simple market maker mirror metrics for the given line."""
    if handle_percent is None:
        raise ValueError(
            "handle_percent is missing in event data. "
            "Do not use ticket_percent as a fallback. "
            "The dataset must contain independent handle and ticket percentages."
        )
    if ticket_percent is None:
        raise ValueError(
            "ticket_percent is missing in event data. "
            "Both handle_percent and ticket_percent must be present."
        )
    from main import MARKET_MAKER_MIRROR_MODEL_PATH
    model_path = model_path or str(MARKET_MAKER_MIRROR_MODEL_PATH)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([
        {
            "opening_odds": price1,
            "handle_percent": handle_percent,
            "ticket_percent": ticket_percent,
            "volatility": 0.0,
        }
    ])
    mirror_price = float(model.predict(df)[0])
    mirror_score = mirror_price - price1
    return {
        "predicted_mirror_price": mirror_price,
        "mirror_score": mirror_score,
    }


def train_market_maker_mirror_model(
    dataset: str, model_out: Optional[str] = None, verbose: bool = False
) -> None:
    """Train a regression model for the market maker mirror and persist it."""
    from main import MARKET_MAKER_MIRROR_MODEL_PATH
    model_out = model_out or str(MARKET_MAKER_MIRROR_MODEL_PATH)
    df = pd.read_csv(dataset)
    required_cols = [
        "opening_odds",
        "handle_percent",
        "ticket_percent",
        "volatility",
        "mirror_target",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[["opening_odds", "handle_percent", "ticket_percent", "volatility"]]
    y = df["mirror_target"]

    if verbose:
        print(
            f"Training regression model on {len(df)} rows with features {X.columns.tolist()}"
        )

    model = LinearRegression()
    model.fit(X, y)

    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    if verbose:
        print(f"Market maker mirror regression model saved to {model_out}")

# Usage example (commented out; use in your CLI or pipeline)
# train_mvp_model("retrosheet_training_data.csv", H2H_MODEL_PATH)
# prob = predict_mvp(H2H_MODEL_PATH, -120, 110)
# print("Win probability:", prob)
