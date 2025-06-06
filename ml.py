import os
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import random
import hashlib

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

##########################################
# --------- Simple File Cache -----------#
##########################################

def _safe_cache_key(*args) -> str:
    """
    Builds a safe filename hash for caching API calls.
    """
    str_key = "-".join(str(x) for x in args)
    return hashlib.md5(str_key.encode()).hexdigest()

def _cache_load(cache_dir: Path, key: str):
    cache_path = cache_dir / f"{key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def _cache_save(cache_dir: Path, key: str, data):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

CACHE_DIR = ROOT_DIR / "api_cache"

##########################################
# --------- API URL + FETCHERS ----------#
##########################################

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

def build_pitcher_ks_url(
    sport_key: str,
    event_id: str,
    *,
    regions: str = "us",
    date_format: str = "iso",
    odds_format: str = "american",
    date: str = None,
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "batter_strikeouts",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if date:
        params["date"] = date
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_historical_games(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
) -> list:
    cache_key = _safe_cache_key("historical", sport_key, date, regions, markets, odds_format)
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

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
            _cache_save(CACHE_DIR, cache_key, data)
            return data
    except urllib.error.HTTPError as e:
        message = e.read().decode() if hasattr(e, "read") else str(e)
        try:
            msg_json = json.loads(message)
            if "INVALID_HISTORICAL_TIMESTAMP" in msg_json.get("error_code", ""):
                print(f"[!] No historical data for {sport_key} on {date}: {msg_json.get('message')}")
                _cache_save(CACHE_DIR, cache_key, [])
                return []
        except Exception:
            pass
        print(f"HTTPError: {message}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []
    except Exception as e:
        print(f"Error fetching historical games: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []

def fetch_pitcher_ks_props(
    sport_key: str,
    event_id: str,
    *,
    date: str,
    regions: str = "us",
    odds_format: str = "american",
) -> list:
    cache_key = _safe_cache_key("ksprops", sport_key, event_id, date, regions, odds_format)
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = build_pitcher_ks_url(
        sport_key,
        event_id,
        regions=regions,
        odds_format=odds_format,
        date=date
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            _cache_save(CACHE_DIR, cache_key, data)
            return data
    except Exception as e:
        _cache_save(CACHE_DIR, cache_key, [])
        print(f"Error fetching pitcher K's props for event {event_id}: {e}")
        return []

##########################################
# ----------- BASELINE LOGIC ------------#
##########################################

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
        games = fetch_historical_games(
            sport_key,
            date=date_str,
            regions=regions,
            markets=markets,
        )
        if verbose:
            print(f"Fetched {len(games)} games for {date_str}")
        for game in games:
            row = _parse_game(game)
            if row:
                rows.append(row)
        current += timedelta(days=1)
    if not rows:
        print(
            "\nNo historical data returned by Odds API for the selected date range.\n"
            "This may be because the data is not yet available for recent games, too old, or for future dates.\n"
            "Try an earlier date range within the last year (MLB data may lag by several days after games are played).\n"
        )
        raise RuntimeError("No historical data returned")
    print(f"Built dataset with {len(rows)} rows.")
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
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")
        start_dt = datetime.fromisoformat(start_date)
        if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
            start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)
        df = build_dataset_from_api(
            sport_key,
            start_dt.strftime("%Y-%m-%d"),
            end_date,
            verbose=verbose,
        )
        train_classifier_df(df, model_out=model_out)
        print(f"Waiting {interval_hours} hours for next training run...")
        time.sleep(interval_hours * 3600)

##########################################
# ---- Incorporation: Pitcher K's Over/Under Prediction Model ----
##########################################

def implied_probability(american_odds: float) -> float:
    # Convert American odds to implied probability
    if american_odds is None:
        return None
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)

def build_ks_dataset_from_api(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a dataset for pitcher K's over/under using K's props and supplement with h2h implied probability.
    """
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    rows = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        games = fetch_historical_games(
            sport_key,
            date=date_str,
            regions=regions,
            markets="h2h",
            odds_format=odds_format,
        )
        if verbose:
            print(f"Fetched {len(games)} games for {date_str}")
        for game in games:
            event_id = game.get("id")
            h2h_home, h2h_away = None, None
            home, away = game.get("home_team"), game.get("away_team")
            for bm in game.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home:
                                h2h_home = outcome.get("price")
                            elif outcome.get("name") == away:
                                h2h_away = outcome.get("price")
            # If no event_id or odds, skip
            if not event_id or (h2h_home is None and h2h_away is None):
                continue
            # Fetch K's O/U market for this event
            ks_markets = fetch_pitcher_ks_props(
                sport_key,
                event_id,
                date=date_str,
                regions=regions,
                odds_format=odds_format,
            )
            # Each bookmaker may offer batter_strikeouts for both teams' pitchers
            for book in ks_markets:
                for market in book.get("markets", []):
                    if market.get("key") == "batter_strikeouts":
                        # Some APIs may structure outcomes differently, so we aggregate by pitcher-line
                        pitcher_lines = {}
                        for outcome in market.get("outcomes", []):
                            pitcher = outcome.get("name")
                            line = outcome.get("line")
                            description = outcome.get("description", "").lower()
                            if pitcher is None or line is None:
                                continue
                            key = (pitcher, line)
                            if key not in pitcher_lines:
                                pitcher_lines[key] = {"pitcher": pitcher, "line": line, "price_over": None, "price_under": None, "over_hit": None}
                            if description.startswith("over"):
                                pitcher_lines[key]["price_over"] = outcome.get("price")
                                pitcher_lines[key]["over_hit"] = 1 if outcome.get("result") == "win" else 0 if outcome.get("result") == "loss" else None
                            elif description.startswith("under"):
                                pitcher_lines[key]["price_under"] = outcome.get("price")
                        # Now combine and add implied win prob
                        for (pitcher, line), props in pitcher_lines.items():
                            implied_win_prob = None
                            if pitcher and home and pitcher in home:
                                implied_win_prob = implied_probability(h2h_home) if h2h_home is not None else None
                            elif pitcher and away and pitcher in away:
                                implied_win_prob = implied_probability(h2h_away) if h2h_away is not None else None
                            if props["price_over"] is not None and props["price_under"] is not None and implied_win_prob is not None and props["over_hit"] is not None:
                                rows.append({
                                    "pitcher": pitcher,
                                    "line": line,
                                    "price_over": props["price_over"],
                                    "price_under": props["price_under"],
                                    "implied_win_prob": implied_win_prob,
                                    "over_hit": props["over_hit"],
                                })
        current += timedelta(days=1)
    if not rows:
        print(
            "\nNo K's O/U data returned by Odds API for the selected date range.\n"
            "Try an earlier date range within the last year.\n"
        )
        raise RuntimeError("No K's O/U data returned")
    print(f"Built K's O/U dataset with {len(rows)} rows.")
    return pd.DataFrame(rows)

def train_pitcher_ks_classifier(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    model_out: str = "pitcher_ks_classifier.pkl",
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> None:
    """
    Train a classifier to predict probability a pitcher goes OVER their K's line,
    using K's props features and h2h implied win probability for context.
    """
    df = build_ks_dataset_from_api(
        sport_key, start_date, end_date, regions=regions, odds_format=odds_format, verbose=verbose
    )
    if verbose:
        print(df.head())
    X = df[["line", "price_over", "price_under", "implied_win_prob"]]
    y = df["over_hit"]
    _train(X, y, model_out)

def predict_pitcher_ks_over_probability(
    model_path: str,
    features: dict,
) -> float:
    """
    Predict the probability a pitcher will go OVER their strikeouts line.
    Features must include: line (float), price_over (float), price_under (float), implied_win_prob (float)
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]
    return float(proba)

##########################################
# ---- END Incorporation ----
##########################################

# CLI entrypoint
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="ML Odds Trainer")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", help="End date for training data (default: today)")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default="moneyline_classifier.pkl")
    parser.add_argument("--once", action="store_true", help="Run only one training (not in a loop)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--ks-incorporation", action="store_true", help="Train pitcher K's O/U classifier using h2h implied win prob as feature")
    args = parser.parse_args()

    end_dt = datetime.utcnow()
    end_date = args.end_date or end_dt.strftime("%Y-%m-%d")
    start_dt = datetime.fromisoformat(args.start_date)
    if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
        start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)

    if args.ks_incorporation:
        # Train the K's O/U model with h2h implied win prob as feature
        train_pitcher_ks_classifier(
            args.sport,
            start_dt.strftime("%Y-%m-%d"),
            end_date,
            model_out="pitcher_ks_classifier.pkl",
            verbose=args.verbose,
        )
    elif args.once:
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
