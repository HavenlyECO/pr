import os
import json
import pickle
import time
import numpy as np
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

# Define this class here so it can be unpickled
class SimpleOddsModel:
    """A model that converts American odds to implied probability."""

    def predict_proba(self, X):
        price1 = X["price1"].values[0]
        if price1 > 0:
            prob = 100 / (price1 + 100)
        else:
            prob = abs(price1) / (abs(price1) + 100)
        return np.array([[1 - prob, prob]])

ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set (check your .env file)")

MAX_HISTORICAL_DAYS = 365

H2H_DATA_DIR = ROOT_DIR / "h2h_data"
H2H_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = H2H_DATA_DIR / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
H2H_MODEL_PATH = H2H_DATA_DIR / "h2h_classifier.pkl"

def to_fixed_utc(date_obj: datetime) -> str:
    """Return ISO-8601 string at fixed 12:00 UTC."""
    return date_obj.strftime("%Y-%m-%dT12:00:00Z")

def _safe_cache_key(*args) -> str:
    str_key = "-".join(str(x) for x in args)
    return hashlib.md5(str_key.encode()).hexdigest()

def _cache_load(cache_dir: Path, key: str):
    cache_path = cache_dir / f"{key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def _cache_save(cache_dir: Path, key: str, data) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

def build_event_ids_url(
    sport_key: str,
    date: str,
    regions: str = "us",
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "h2h",
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_event_ids_historical(
    sport_key: str,
    date: str,
    regions: str = "us",
) -> list:
    """Fetch all event IDs for a given sport and date (historical snapshot)."""
    url = build_event_ids_url(sport_key, date, regions)
    cache_key = _safe_cache_key("eventids", sport_key, date, regions, "h2h")
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, list):
            raise ValueError(f"Unexpected event ids response: {data!r}")
        event_ids = []
        for g in data:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            for book in g.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == "h2h":
                        event_ids.append(g["id"])
                        break
                else:
                    continue
                break
        _cache_save(CACHE_DIR, cache_key, event_ids)
        return event_ids
    except Exception as e:
        print(f"Error fetching event ids for {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_h2h_event_ids_url(
    sport_key: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
) -> str:
    """Build URL for head-to-head event IDs."""
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_h2h_event_ids(
    sport_key: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
    verbose: bool = False,
) -> list:
    """Fetch event IDs using the provided API key."""
    url = build_h2h_event_ids_url(
        sport_key, date, api_key=api_key, regions=regions
    )
    cache_key = _safe_cache_key(
        "h2h_event_ids", sport_key, date, regions, api_key
    )
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, list):
            raise ValueError(f"Unexpected event ids response: {data!r}")
        event_ids = []
        for g in data:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            for book in g.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == "h2h":
                        event_ids.append(g["id"])
                        break
                else:
                    continue
                break
        _cache_save(CACHE_DIR, cache_key, event_ids)
        if verbose:
            print(f"Fetched {len(event_ids)} event ids for {date}")
        return event_ids
    except Exception as e:
        print(f"Error fetching h2h event ids for {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_historical_odds_url(
    sport_key: str,
    date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "h2h",
        "date": date,
        "oddsFormat": odds_format,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_historical_h2h_odds(
    sport_key: str,
    date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
) -> list:
    """Return all h2h odds for a sport and date."""
    url = build_historical_odds_url(
        sport_key, date, regions=regions, odds_format=odds_format
    )
    cache_key = _safe_cache_key(
        "historicalodds", sport_key, date, regions, odds_format
    )
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())

        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "data" in data:
            events = data["data"]
            if not isinstance(events, list):
                raise ValueError(
                    f"Unexpected historical odds response: {data!r}"
                )
        else:
            raise ValueError(f"Unexpected historical odds response: {data!r}")

        _cache_save(CACHE_DIR, cache_key, events)
        return events
    except Exception as e:
        print(f"Error fetching historical odds for {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []

def build_h2h_url_historical(
    sport_key: str,
    event_id: str,
    date: str,
    regions: str = "us",
    date_format: str = "iso",
    odds_format: str = "american",
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_h2h_props_historical(
    sport_key: str,
    event_id: str,
    date: str,
    regions: str = "us",
    odds_format: str = "american",
) -> list:
    cache_key = _safe_cache_key("h2hprops", sport_key, event_id, date, regions, odds_format)
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = build_h2h_url_historical(
        sport_key, event_id, date=date, regions=regions, odds_format=odds_format
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())

            out: list
            if isinstance(data, dict):
                if "bookmakers" in data:
                    out = data["bookmakers"]
                elif "data" in data and isinstance(data["data"], dict):
                    out = data["data"].get("bookmakers", [])
                else:
                    out = []
            else:
                out = []

            _cache_save(CACHE_DIR, cache_key, out)
            return out
    except Exception as e:
        print(f"Error fetching h2h props for event {event_id} on {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_h2h_props_url(
    sport_key: str,
    event_id: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
    date_format: str = "iso",
    odds_format: str = "american",
) -> str:
    """Build URL for head-to-head props."""
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    )
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_h2h_props(
    sport_key: str,
    event_id: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> list:
    """Fetch h2h props for the given event ID."""
    cache_key = _safe_cache_key(
        "h2h_props", sport_key, event_id, date, regions, odds_format, api_key
    )
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = build_h2h_props_url(
        sport_key,
        event_id,
        date,
        api_key=api_key,
        regions=regions,
        odds_format=odds_format,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, dict) and "bookmakers" in data:
                out = data["bookmakers"]
            else:
                out = []
            _cache_save(CACHE_DIR, cache_key, out)
            if verbose:
                print(
                    f"Fetched props for event {event_id} on {date} ({len(out)} bookmakers)"
                )
            return out
    except Exception as e:
        print(f"Error fetching h2h props for event {event_id} on {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []

def build_h2h_dataset_from_api(
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
        date_str = to_fixed_utc(current)
        events = fetch_historical_h2h_odds(
            sport_key,
            date_str,
            regions=regions,
            odds_format=odds_format,
        )
        if verbose:
            print(f"Fetched {len(events)} events for {date_str}")
        for game in events:
            if not isinstance(game, dict):
                continue
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) != 2:
                        continue
                    team1 = outcomes[0].get("name")
                    team2 = outcomes[1].get("name")
                    price1 = outcomes[0].get("price")
                    price2 = outcomes[1].get("price")
                    result1 = outcomes[0].get("result")
                    result2 = outcomes[1].get("result")
                    if None in (team1, team2, price1, price2, result1, result2):
                        continue
                    if verbose:
                        print(
                            f"DEBUG: {team1} vs {team2} | price1={price1}, price2={price2}, result1={result1}, result2={result2}"
                        )
                    label = 1 if result1 == "win" else 0
                    rows.append({
                        "team1": team1,
                        "team2": team2,
                        "price1": price1,
                        "price2": price2,
                        "team1_win": label,
                    })
                    break
        current += timedelta(days=1)
    if not rows:
        raise RuntimeError("No h2h data returned")
    if verbose:
        print(f"Built h2h dataset with {len(rows)} rows.")
    return pd.DataFrame(rows)

def _train(X: pd.DataFrame, y: pd.Series, model_out: str) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {out_path}")

def train_h2h_classifier(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    model_out: str = str(H2H_MODEL_PATH),
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> None:
    df = build_h2h_dataset_from_api(
        sport_key,
        start_date,
        end_date,
        regions=regions,
        odds_format=odds_format,
        verbose=verbose,
    )
    if verbose:
        print(df.head())
    X = df[["price1", "price2"]]
    y = df["team1_win"]
    _train(X, y, model_out)

def predict_h2h_probability(
    model_path: str,
    price1: float,
    price2: float,
) -> float:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([{"price1": price1, "price2": price2}])
    proba = model.predict_proba(df)[0][1]
    return float(proba)

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train a head-to-head classifier using historical odds endpoint")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model-out", default=str(H2H_MODEL_PATH))
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    train_h2h_classifier(
        args.sport,
        args.start_date,
        args.end_date,
        model_out=args.model_out,
        verbose=args.verbose,
    )


def demo_fetch() -> None:
    """Example usage of fetching h2h data."""
    event_ids = fetch_h2h_event_ids(
        sport_key="baseball_mlb",
        date="2025-06-01T12:00:00Z",
        api_key=API_KEY,
        verbose=True,
    )

    for event_id in event_ids:
        bookmakers = fetch_h2h_props(
            sport_key="baseball_mlb",
            event_id=event_id,
            date="2025-06-01T12:00:00Z",
            api_key=API_KEY,
            verbose=True,
        )
        # process bookmakers as before...

if __name__ == "__main__":
    _cli()
    demo_fetch()
