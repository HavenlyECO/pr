import os
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import urllib.error
import hashlib
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(funcName)s:%(lineno)d: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ml_debug")

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required. Install it with 'pip install python-dotenv'")

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


def to_pst_iso8601(date_obj: datetime) -> str:
    """Return ISO-8601 date string at 12:00 UTC."""
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


def build_h2h_url(
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
        "markets": "h2h",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if date:
        params["date"] = date
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_h2h_props(
    sport_key: str,
    event_id: str,
    *,
    date: str,
    regions: str = "us",
    odds_format: str = "american",
) -> list:
    cache_key = _safe_cache_key("h2hprops", sport_key, event_id, date, regions, odds_format)
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = build_h2h_url(
        sport_key,
        event_id,
        regions=regions,
        odds_format=odds_format,
        date=date,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, (list, dict)):
                _cache_save(CACHE_DIR, cache_key, data)
            return data
    except Exception as e:
        print(f"Error fetching h2h props for event {event_id}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def fetch_h2h_event_ids(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
) -> list:
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
        f"?apiKey={API_KEY}&regions={regions}&date={date}&markets=h2h"
    )
    cache_key = _safe_cache_key("eventids", sport_key, date, regions, "h2h")
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())

        games = data.get("data") if isinstance(data, dict) and "data" in data else data
        if not isinstance(games, list):
            raise ValueError(f"Unexpected event ids response: {games!r}")

        event_ids: list[str] = []
        for g in games:
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


def build_h2h_dataset_from_api(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> pd.DataFrame:
    import collections
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    rows: list[dict] = []
    missing_results = collections.defaultdict(list)  # event_id -> [(teams, date, outcomes)]
    current = start
    while current <= end:
        date_str = to_pst_iso8601(current)
        event_ids = fetch_h2h_event_ids(
            sport_key,
            date=date_str,
            regions=regions,
        )
        if verbose:
            print(f"Fetched {len(event_ids)} event ids for {date_str}")
        for event_id in event_ids:
            h2h_markets = fetch_h2h_props(
                sport_key,
                event_id,
                date=date_str,
                regions=regions,
                odds_format=odds_format,
            )
            for book in h2h_markets:
                if not isinstance(book, dict):
                    continue
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
                    if None in (result1, result2):
                        if event_id not in missing_results:
                            missing_results[event_id].append(
                                (f"{team1} vs {team2}", date_str, outcomes)
                            )
                        break
                    if None in (team1, team2, price1, price2):
                        continue
                    label = 1 if result1 == "win" else 0
                    rows.append(
                        {
                            "team1": team1,
                            "team2": team2,
                            "price1": price1,
                            "price2": price2,
                            "team1_win": label,
                        }
                    )
                    break
        current += timedelta(days=1)

    if missing_results:
        print("\nSummary of events with missing results (not used for training):")
        for event_id, infos in missing_results.items():
            for (teams, date_str, outcomes) in infos:
                print(f"  - Event {event_id} | {teams} | Date: {date_str}")
                print(f"    Outcomes: {json.dumps(outcomes, indent=2)}")
        print(f"Total events with missing results: {len(missing_results)}\n")

    if not rows:
        print(
            "[ERROR] No h2h data returned. All events are missing results fields. Try querying a range of dates with only historical/settled games."
        )
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
    preds = model.predict(X_test)
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


# ==================== CLI entrypoint ====================

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train a head-to-head classifier")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", help="End date for training data (default: today)")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default=str(H2H_MODEL_PATH))
    parser.add_argument("--once", action="store_true", help="Run only one training (not in a loop)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    end_dt = datetime.utcnow()
    end_date = args.end_date or end_dt.strftime("%Y-%m-%d")
    start_dt = datetime.fromisoformat(args.start_date)
    if (end_dt - start_dt).days > MAX_HISTORICAL_DAYS:
        start_dt = end_dt - timedelta(days=MAX_HISTORICAL_DAYS)

    if args.once:
        train_h2h_classifier(
            args.sport,
            start_dt.strftime("%Y-%m-%d"),
            end_date,
            model_out=args.model_out,
            verbose=args.verbose,
        )
    else:
        while True:
            train_h2h_classifier(
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
