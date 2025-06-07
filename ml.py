import os
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import urllib.error
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

# --------- Simple File Cache -----------#

def _safe_cache_key(*args) -> str:
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


def build_pitcher_ks_url(
    sport_key: str,
    event_id: str,
    *,
    regions: str = "us",
    date_format: str = "iso",
    odds_format: str = "american",
    date: str | None = None,
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
        date=date,
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


def fetch_all_event_ids(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
) -> list:
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
        f"?apiKey={API_KEY}&regions={regions}&date={date}"
    )
    cache_key = _safe_cache_key("eventids", sport_key, date, regions)
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    print(f"[DEBUG] Fetching event IDs URL: {url}")
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            event_ids = [game.get("id") for game in data if game.get("id")]
            _cache_save(CACHE_DIR, cache_key, event_ids)
            return event_ids
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTPError for event ids on {date}: {e.code} {e.reason}")
        if hasattr(e, "read"):
            error_body = e.read()
            print(f"[ERROR] Error body: {error_body.decode(errors='replace')}")
        print(f"[ERROR] URL was: {url}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []
    except Exception as e:
        print(f"[ERROR] General error fetching event ids for {date}: {e}")
        print(f"[ERROR] URL was: {url}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_ks_dataset_from_api(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> pd.DataFrame:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    rows: list[dict] = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        event_ids = fetch_all_event_ids(
            sport_key,
            date=date_str,
            regions=regions,
        )
        if verbose:
            print(f"Fetched {len(event_ids)} event ids for {date_str}")
        for event_id in event_ids:
            ks_markets = fetch_pitcher_ks_props(
                sport_key,
                event_id,
                date=date_str,
                regions=regions,
                odds_format=odds_format,
            )
            for book in ks_markets:
                for market in book.get("markets", []):
                    if market.get("key") == "batter_strikeouts":
                        pitcher_lines: dict[tuple, dict] = {}
                        for outcome in market.get("outcomes", []):
                            pitcher = outcome.get("name")
                            line = outcome.get("line")
                            description = outcome.get("description", "").lower()
                            if pitcher is None or line is None:
                                continue
                            key = (pitcher, line)
                            if key not in pitcher_lines:
                                pitcher_lines[key] = {
                                    "pitcher": pitcher,
                                    "line": line,
                                    "price_over": None,
                                    "price_under": None,
                                    "over_hit": None,
                                }
                            if description.startswith("over"):
                                pitcher_lines[key]["price_over"] = outcome.get("price")
                                pitcher_lines[key]["over_hit"] = (
                                    1
                                    if outcome.get("result") == "win"
                                    else 0
                                    if outcome.get("result") == "loss"
                                    else None
                                )
                            elif description.startswith("under"):
                                pitcher_lines[key]["price_under"] = outcome.get("price")
                        for props in pitcher_lines.values():
                            if (
                                props["price_over"] is not None
                                and props["price_under"] is not None
                                and props["over_hit"] is not None
                            ):
                                rows.append(props)
        current += timedelta(days=1)

    if not rows:
        print(
            "\nNo K's O/U data returned by Odds API for the selected date range.\n"
            "Try an earlier date range within the last year.\n"
        )
        raise RuntimeError("No K's O/U data returned")
    if verbose:
        print(f"Built K's O/U dataset with {len(rows)} rows.")
    return pd.DataFrame(rows)


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
    df = build_ks_dataset_from_api(
        sport_key,
        start_date,
        end_date,
        regions=regions,
        odds_format=odds_format,
        verbose=verbose,
    )
    if verbose:
        print(df.head())
    X = df[["line", "price_over", "price_under"]]
    y = df["over_hit"]
    _train(X, y, model_out)


def predict_pitcher_ks_over_probability(
    model_path: str,
    features: dict,
) -> float:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]
    return float(proba)


# ==================== CLI entrypoint ====================

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="ML Odds Trainer")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", help="End date for training data (default: today)")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default="pitcher_ks_classifier.pkl")
    parser.add_argument("--once", action="store_true", help="Run only one training (not in a loop)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    end_dt = datetime.utcnow()
    end_date = args.end_date or end_dt.strftime("%Y-%m-%d")
    start_dt = datetime.fromisoformat(args.start_date)
    if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
        start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)

    if args.once:
        train_pitcher_ks_classifier(
            args.sport,
            start_dt.strftime("%Y-%m-%d"),
            end_date,
            model_out=args.model_out,
            verbose=args.verbose,
        )
    else:
        while True:
            train_pitcher_ks_classifier(
                args.sport,
                start_dt.strftime("%Y-%m-%d"),
                end_date,
                model_out=args.model_out,
                verbose=args.verbose,
            )
            print(f"Waiting {args.interval_hours} hours for next training run...")
            time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    _cli()
