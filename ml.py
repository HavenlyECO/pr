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
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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


def line_movement_delta(closing_odds: float, opening_odds: float) -> float:
    """Return the change in odds from open to close."""
    return closing_odds - opening_odds

def compute_recency_weights(dates: pd.Series, half_life_days: float = 30.0) -> pd.Series:
    """Return exponentially decayed weights based on recency.

    The most recent date receives weight 1.0 and each ``half_life_days`` offset
    halves the weight. Missing dates result in uniform weight 1.0.
    """
    if dates.isnull().all():
        return pd.Series(1.0, index=dates.index)
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.isnull().all():
        return pd.Series(1.0, index=dates.index)
    latest = parsed.max()
    age = (latest - parsed).dt.days.fillna(0)
    weights = 0.5 ** (age / float(half_life_days))
    return weights

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
MONEYLINE_MODEL_PATH = ROOT_DIR / "moneyline_classifier.pkl"

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
                        "implied_prob": american_odds_to_prob(price1),
                        "team1_win": label,
                        "event_date": current.date().isoformat(),
                    })
                    break
        current += timedelta(days=1)
    if not rows:
        raise RuntimeError("No h2h data returned")
    if verbose:
        print(f"Built h2h dataset with {len(rows)} rows.")
    return pd.DataFrame(rows)

def _train(
    X: pd.DataFrame,
    y: pd.Series,
    model_out: str,
    sample_weight: pd.Series | None = None,
) -> None:
    """Train a logistic regression model with feature normalization.

    Continuous inputs are standardized using ``StandardScaler`` before
    fitting to improve convergence. Probabilities are calibrated with
    isotonic regression on a validation split.
    This approach keeps modeling in a regression setup so you can choose any probability threshold later for classification.
    """

    if sample_weight is not None:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, sample_weight, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if sample_weight is not None:
        pipeline.fit(X_train, y_train, sample_weight=w_train)
    else:
        pipeline.fit(X_train, y_train)

    # Calibrate using isotonic regression on the validation set
    calibrator = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
    if sample_weight is not None:
        calibrator.fit(X_val, y_val, sample_weight=w_val)
    else:
        calibrator.fit(X_val, y_val)

    probas = calibrator.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probas)
    brier = brier_score_loss(y_val, probas)
    print(f"Validation AUC: {auc:.3f}, Brier score: {brier:.3f}")

    def _report_segment_metrics() -> None:
        if not any(col.startswith("live_inning_") for col in X_val.columns):
            return

        segments = {
            "pregame": X_val[X_val.filter(like="live_inning_").isna().all(axis=1)],
            "5th inning": X_val[X_val.get("live_inning_5_diff").notna()],
            "7th inning": X_val[X_val.get("live_inning_7_diff").notna()],
        }

        for name, seg_X in segments.items():
            if seg_X.empty:
                continue
            seg_y = y_val.loc[seg_X.index]
            seg_proba = calibrator.predict_proba(seg_X)[:, 1]
            seg_auc = roc_auc_score(seg_y, seg_proba)
            seg_brier = brier_score_loss(seg_y, seg_proba)
            print(f"  {name} AUC: {seg_auc:.3f}, Brier: {seg_brier:.3f}")

    _report_segment_metrics()

    residuals_df = pd.DataFrame({
        "true_label": y_val.reset_index(drop=True),
        "model_prob": probas,
    })
    residuals_df["residual"] = residuals_df["true_label"] - residuals_df["model_prob"]

    residuals_path = Path(model_out).with_suffix(".residuals.csv")
    residuals_df.to_csv(residuals_path, index=False)
    print(f"Residuals saved to {residuals_path}")

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(calibrator, f)

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
    recent_half_life: float | None = None,
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
    weights = None
    if recent_half_life is not None and "event_date" in df.columns:
        weights = compute_recency_weights(df["event_date"], half_life_days=recent_half_life)
    _train(X, y, model_out, sample_weight=weights)

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


def split_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate pregame and live-game feature DataFrames.

    Any columns containing post-result information are discarded to avoid data
    leakage when training or predicting.
    """

    disallow_keywords = {"result", "final", "post", "future"}

    def allowed(col: str) -> bool:
        c_lower = col.lower()
        return not any(k in c_lower for k in disallow_keywords)

    pregame_cols = [c for c in df.columns if c.startswith("pregame_") and allowed(c)]
    live_cols = [c for c in df.columns if c.startswith("live_") and allowed(c)]

    # Ignore any columns that might leak future information
    other_cols = [
        c
        for c in df.columns
        if c not in pregame_cols + live_cols and allowed(c)
    ]

    pregame_df = df[pregame_cols + other_cols].copy()
    live_df = df[live_cols].copy()
    return pregame_df, live_df


def train_moneyline_classifier(
    dataset_path: str,
    *,
    model_out: str = str(MONEYLINE_MODEL_PATH),
    features_type: str = "pregame",
    verbose: bool = False,
    recent_half_life: float | None = None,
    date_column: str | None = None,
) -> None:
    """Train a logistic regression model from a CSV with a home_team_win column.

Modeling is done in regression mode first. You can later apply a probability threshold for classification if desired."""
    df = pd.read_csv(dataset_path)
    if "home_team_win" not in df.columns:
        raise ValueError("Dataset must include 'home_team_win' column")

    # Automatically create a line movement feature if possible
    if {
        "opening_odds",
        "closing_odds",
    }.issubset(df.columns):
        df["line_delta"] = df["closing_odds"] - df["opening_odds"]
        if verbose:
            print("Computed line_delta column from closing and opening odds")

    features_df = df.drop(columns=["home_team_win"])
    pregame_X, live_X = split_feature_sets(features_df)
    X = live_X if features_type == "live" else pregame_X
    y = df["home_team_win"]
    if verbose:
        print(
            f"Training dataset with {len(df)} rows using {features_type} features ({len(X.columns)} columns)"
        )

    if X.empty:
        raise ValueError(f"No columns found for feature type: {features_type}")

    weights = None
    if recent_half_life is not None:
        col = (
            date_column
            if date_column and date_column in df.columns
            else next((c for c in df.columns if "date" in c.lower()), None)
        )
        if col is not None:
            weights = compute_recency_weights(
                pd.to_datetime(df[col], errors="coerce"),
                half_life_days=recent_half_life,
            )
            if verbose:
                print(f"Using column '{col}' for recency weighting")

    _train(X, y, model_out, sample_weight=weights)


def predict_moneyline_probability(
    model_path: str,
    features: dict,
) -> float:
    """Predict win probability using a trained moneyline classifier."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([features])
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
