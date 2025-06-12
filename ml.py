import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import pickle
import warnings

# Access model path constants from main without creating a circular import
from typing import Optional
from odds_utils import american_odds_to_prob, american_odds_to_payout

# Functions only; no code at global scope except imports and definitions.

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


def extract_market_signals(event):
    """
    Given a single event (dict), return features for the mirror model:
    - opening_odds
    - volatility
    (closing_odds is only available after event completion)
    """
    opening_odds = event.get("opening_price", event.get("price"))
    volatility = event.get("volatility")
    if opening_odds is None or volatility is None:
        raise ValueError(
            "Missing required features for market maker mirror model (opening_odds or volatility)."
        )
    return {
        "opening_odds": opening_odds,
        "volatility": volatility,
    }


def market_maker_mirror_score(model_path: str, features: dict, current_odds: float) -> float:
    """Return the difference between a mirrored closing price and ``current_odds``.

    Parameters
    ----------
    model_path : str
        Path to the trained regression model produced by
        :func:`train_market_maker_mirror_model`.
    features : dict
        Mapping with at least ``opening_odds`` and ``volatility`` keys as
        produced by :func:`extract_market_signals`.
    current_odds : float
        The team's current moneyline odds.

    Returns
    -------
    float
        The mirrored score. Positive when the projected closing price is longer
        than ``current_odds`` and negative when shaded.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = [[features["opening_odds"], features["volatility"]]]
    projected = model.predict(X)[0]
    return float(projected - current_odds)


def train_market_maker_mirror_model(dataset, model_out, verbose=False):
    """
    Train a regression model for the market maker mirror and save it to model_out.
    """
    df = pd.read_csv(dataset)
    # Either use line_move as the target or closing_odds; adjust as needed
    required_cols = ["opening_odds", "closing_odds", "volatility", "mirror_target"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset {dataset} missing required columns: {missing}")

    X = df[["opening_odds", "volatility"]]
    y = df["mirror_target"]

    if verbose:
        print(f"Training regression model on {len(df)} rows with features {X.columns.tolist()}")

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
