import os
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import logging

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv is required. Install it with 'pip install python-dotenv'"
    )

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Explicitly load environment variables from the repo root .env
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if not DOTENV_PATH.exists():
    raise RuntimeError(f".env file not found at {DOTENV_PATH}")
load_dotenv(dotenv_path=DOTENV_PATH)

API_KEY = os.getenv("THE_ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "THE_ODDS_API_KEY environment variable is not set (check /pr/.env)"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

# The Odds API only allows fetching historical data for roughly the last year.
# Requests outside this window return HTTP 422 (INVALID_HISTORICAL_TIMESTAMP).
MAX_HISTORICAL_DAYS = 365


def build_historical_odds_url(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    include_scores: bool = False,
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
    if include_scores:
        params["include"] = "scores"
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
        include_scores=True,
    )
    logging.info("Fetching %s", url)
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            logging.info("Fetched %d games for %s", len(data), date)
            return data
    except urllib.error.HTTPError as e:  # pragma: no cover - network error handling
        message = e.read().decode() if hasattr(e, "read") else str(e)
        if "INVALID_HISTORICAL_TIMESTAMP" in message:
            logging.warning("Historical data is not available for date %s", date)
            raise ValueError(
                "Historical data is not available for the requested date"
            ) from e
        logging.error("Failed to fetch historical games for %s: %s", date, message)
        raise RuntimeError(f"Failed to fetch historical games: {message}") from e


def _parse_game(game: dict) -> dict | None:
    """Extract training features from a game record."""
    home = game.get("home_team")
    away = game.get("away_team")
    if not home or not away:
        logging.debug("Missing team info, skipping game.")
        return None
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
        logging.debug("Missing price info, skipping game.")
        return None
    if home not in scores or away not in scores:
        logging.debug("Missing score info, skipping game.")
        return None
    try:
        home_score = float(scores[home])
        away_score = float(scores[away])
    except Exception:
        logging.debug("Score is not numeric, skipping game.")
        return None

    home_win = 1 if home_score > away_score else 0
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
    if (end - start).days > MAX_HISTORICAL_DAYS:
        raise ValueError(
            f"Date range exceeds {MAX_HISTORICAL_DAYS} days which is the maximum allowed by the Odds API"
        )
    rows: list[dict] = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            games = fetch_historical_games(
                sport_key,
                date=date_str,
                regions=regions,
                markets=markets,
            )
        except ValueError:
            # Skip dates that are outside the API's historical range
            current += timedelta(days=1)
            continue

        if not games:
            logging.info("No games returned for %s", date_str)
        else:
            logging.info("Fetched %d games for %s", len(games), date_str)

        for game in games:
            row = _parse_game(game)
            if not row:
                logging.debug("Skipping game, parse failed")
            else:
                rows.append(row)
        current += timedelta(days=1)
    if not rows:
        print(
            "\nNo historical data returned by Odds API for the selected date range.\n"
            "This may be because the data is not yet available for recent games or for future dates.\n"
            "Try an earlier date range (at least a week or two in the past).\n"
        )
        raise RuntimeError("No historical data returned")
    logging.info("Dataset contains %d rows.", len(rows))
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
    logging.info("Validation accuracy: %.3f", acc)
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved to %s", model_out)


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


def continuous_train_classifier(
    sport_key: str,
    start_date: str,
    *,
    interval_hours: int = 24,
    model_out: str = "moneyline_classifier.pkl",
) -> None:
    """Continuously train the classifier on new data.

    This utility repeatedly fetches historical odds from ``start_date`` up to
    the current date, trains a new model and saves it to ``model_out`` every
    ``interval_hours`` hours. The loop runs indefinitely until interrupted.
    """

    while True:
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")

        # The Odds API only supports a one year historical window. If the
        # configured ``start_date`` exceeds this range, clamp it so that the
        # request stays within the allowed limit.
        start_dt = datetime.fromisoformat(start_date)
        if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
            start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)

        df = build_dataset_from_api(
            sport_key,
            start_dt.strftime("%Y-%m-%d"),
            end_date,
        )
        train_classifier_df(df, model_out=model_out)
        logging.info("Waiting %d hours for next training run...", interval_hours)
        time.sleep(interval_hours * 3600)


def _cli() -> None:
    """Command line interface for training utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="ML Odds Trainer")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default="moneyline_classifier.pkl")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only one training cycle instead of looping",
    )
    args = parser.parse_args()

    if args.once:
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")
        start_dt = datetime.fromisoformat(args.start_date)
        if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
            start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)
        df = build_dataset_from_api(
            args.sport,
            start_dt.strftime("%Y-%m-%d"),
            end_date,
        )
        train_classifier_df(df, model_out=args.model_out)
    else:
        continuous_train_classifier(
            args.sport,
            args.start_date,
            interval_hours=args.interval_hours,
            model_out=args.model_out,
        )


if __name__ == "__main__":  # pragma: no cover - CLI execution
    _cli()
