import os
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import random

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required. Install it with 'pip install python-dotenv'")

# Always load .env from the project root
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)
else:
    print(f"Warning: .env file not found at {DOTENV_PATH}")

API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set (check your .env file)")

MAX_HISTORICAL_DAYS = 365


def safe_fromisoformat(dtstr: str) -> datetime:
    """Parse an ISO 8601 string, handling trailing 'Z' or 'z' for UTC."""
    if dtstr.endswith("z") or dtstr.endswith("Z"):
        dtstr = dtstr[:-1] + "+00:00"
    return datetime.fromisoformat(dtstr)

def build_historical_odds_url(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    include_scores: bool = False,
) -> str:
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
    url = build_historical_odds_url(
        sport_key,
        date=date,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        include_scores=True,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            # Always extract the "data" key if present
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data if isinstance(data, list) else []
    except urllib.error.HTTPError as e:
        message = e.read().decode() if hasattr(e, "read") else str(e)
        try:
            msg_json = json.loads(message)
            if "INVALID_HISTORICAL_TIMESTAMP" in msg_json.get("error_code", ""):
                print(f"[!] No historical data for {sport_key} on {date}: {msg_json.get('message')}")
                return []
        except Exception:
            pass
        print(f"HTTPError: {message}")
        return []
    except Exception as e:
        print(f"Error fetching historical games: {e}")
        return []

def _parse_game(game: dict) -> dict | None:
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
    verbose: bool = False,
) -> pd.DataFrame:
    print(
        f"[INFO] Building dataset from API for {sport_key} from {start_date} to {end_date} ..."
    )
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    if (end - start).days > MAX_HISTORICAL_DAYS:
        raise ValueError(
            f"Date range exceeds {MAX_HISTORICAL_DAYS} days which is the maximum allowed by the Odds API"
        )
    rows: list[dict] = []
    current = start
    while current <= end:
        # The API expects an ISO 8601 timestamp. Use a fixed time (noon UTC)
        # to avoid timezone issues when only a date is provided.
        date_iso = current.strftime("%Y-%m-%dT12:00:00Z")
        print(f"  [FETCH] Fetching games for {date_iso} ...")
        games = fetch_historical_games(
            sport_key,
            date=date_iso,
            regions=regions,
            markets=markets,
        )
        print(f"  [FETCH] {len(games)} games fetched for {date_iso}")
        for game in games:
            row = _parse_game(game)
            if row:
                rows.append(row)
        current += timedelta(days=1)
    if not rows:
        print(
            "\n[WARNING] No historical data returned by Odds API for the selected date range.\n"
            "This may be because the data is not yet available for recent games, too old, or for future dates.\n"
            "Try an earlier date range within the last year.\n"
        )
        raise RuntimeError("No historical data returned")
    print(f"[INFO] Built dataset with {len(rows)} rows.")
    return pd.DataFrame(rows)

def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "home_team_win" not in df.columns:
        raise ValueError("Dataset must contain 'home_team_win' column")
    X = df.drop(columns=["home_team_win"])
    y = df["home_team_win"]
    return X, y

def _train(X: pd.DataFrame, y: pd.Series, model_out: str) -> None:
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
    X, y = load_dataset(dataset_path)
    _train(X, y, model_out)

def train_classifier_df(df: pd.DataFrame, model_out: str = "moneyline_classifier.pkl") -> None:
    if "home_team_win" not in df.columns:
        raise ValueError("DataFrame missing 'home_team_win' column")
    X = df.drop(columns=["home_team_win"])
    y = df["home_team_win"]
    _train(X, y, model_out)

def predict_win_probability(model_path: str, features: dict) -> float:
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
    verbose: bool = False,
) -> None:
    while True:
        print("\n[INFO] Starting new training cycle ...")
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")
        start_dt = datetime.fromisoformat(start_date)
        if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
            start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)
        print(
            f"[INFO] Training window: {start_dt.date()} to {end_date} (UTC now: {end_dt.isoformat()})"
        )
        try:
            df = build_dataset_from_api(
                sport_key,
                start_dt.strftime("%Y-%m-%d"),
                end_date,
                verbose=verbose,
            )
            print(f"[INFO] Training classifier and saving to {model_out} ...")
            train_classifier_df(df, model_out=model_out)
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
        print(f"[INFO] Waiting {interval_hours} hours for next training run...")
        time.sleep(interval_hours * 3600)

# CLI entrypoint
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="ML Odds Trainer")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default="moneyline_classifier.pkl")
    parser.add_argument("--once", action="store_true", help="Run only one training (not in a loop)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.once:
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")
        start_dt = safe_fromisoformat(args.start_date)
        if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
            start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)
        df = build_dataset_from_api(args.sport, start_dt.strftime("%Y-%m-%d"), end_date, verbose=args.verbose)
        train_classifier_df(df, model_out=args.model_out)
    else:
        continuous_train_classifier(
            args.sport,
            args.start_date,
            interval_hours=args.interval_hours,
            model_out=args.model_out,
            verbose=args.verbose,
        )

if __name__ == "__main__":
    _cli()
