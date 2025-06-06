import os
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


API_KEY = os.getenv("THE_ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set")


def build_historical_odds_url(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
) -> str:
    """Return historical odds API URL."""
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_historical_games(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
) -> list:
    """Return historical games for a single date."""
    url = build_historical_odds_url(
        sport_key,
        date=date,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
    )
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def _parse_game(game: dict) -> dict | None:
    """Extract training features from a game record."""
    home = game.get("home_team")
    away = game.get("away_team")
    home_price = None
    away_price = None
    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home:
                        home_price = outcome.get("price")
                    elif outcome.get("name") == away:
                        away_price = outcome.get("price")
                if home_price is not None and away_price is not None:
                    break
        if home_price is not None and away_price is not None:
            break

    scores = {s.get("name"): s.get("score") for s in (game.get("scores") or [])}
    if home_price is None or away_price is None:
        return None
    if home not in scores or away not in scores:
        return None
    home_win = 1 if scores[home] > scores[away] else 0
    return {
        "home_price": home_price,
        "away_price": away_price,
        "home_team_win": home_win,
    }


def build_dataset_from_api(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    regions: str = "us",
    markets: str = "h2h",
) -> pd.DataFrame:
    """Return a dataframe of historical games between ``start_date`` and ``end_date``."""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    rows: list[dict] = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        games = fetch_historical_games(
            sport_key,
            date=date_str,
            regions=regions,
            markets=markets,
        )
        for game in games:
            row = _parse_game(game)
            if row:
                rows.append(row)
        current += timedelta(days=1)
    if not rows:
        raise RuntimeError("No historical data returned")
    return pd.DataFrame(rows)


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV file.

    The CSV must contain a ``home_team_win`` column which is used as the target
    label (1 for home team win, 0 for away team win).
    All other columns are treated as features.
    """
    df = pd.read_csv(path)
    if "home_team_win" not in df.columns:
        raise ValueError("Dataset must contain 'home_team_win' column")
    X = df.drop(columns=["home_team_win"])
    y = df["home_team_win"]
    return X, y


def _train(X: pd.DataFrame, y: pd.Series, model_out: str) -> None:
    """Internal helper to fit logistic regression and persist the model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_out}")


def train_classifier(dataset_path: str, model_out: str = "moneyline_classifier.pkl") -> None:
    """Train a classifier from a CSV dataset."""
    X, y = load_dataset(dataset_path)
    _train(X, y, model_out)


def train_classifier_df(df: pd.DataFrame, model_out: str = "moneyline_classifier.pkl") -> None:
    """Train a classifier from a dataframe returned by ``build_dataset_from_api``."""
    if "home_team_win" not in df.columns:
        raise ValueError("DataFrame missing 'home_team_win' column")
    X = df.drop(columns=["home_team_win"])
    y = df["home_team_win"]
    _train(X, y, model_out)


def predict_win_probability(model_path: str, features: dict) -> float:
    """Predict probability of the home team winning using the trained model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]
    return float(proba)
