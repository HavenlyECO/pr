#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json
import logging
import warnings

LOG_FILE = Path("training_warnings.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logging.captureWarnings(True)

# Import from your ml.py module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml import H2H_DATA_DIR, H2H_MODEL_PATH, CACHE_DIR, american_odds_to_prob

# Define at module level (not inside a function) so it can be pickled
class SimpleOddsModel:
    """A model that converts American odds to implied probability."""

    def predict_proba(self, X):
        price1 = X["price1"].values[0]
        if price1 > 0:
            prob = 100 / (price1 + 100)
        else:
            prob = abs(price1) / (abs(price1) + 100)
        return np.array([[1 - prob, prob]])


def examine_cache_files(cache_dir=CACHE_DIR, max_files=5):
    """Print a detailed look at cached API responses for debugging."""

    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        msg = f"No cache files found in {cache_dir}"
        print(msg)
        logging.warning(msg)
        return

    print(
        f"Found {len(cache_files)} cache files. "
        f"Examining {min(max_files, len(cache_files))} files in detail."
    )

    for i, file_path in enumerate(cache_files[:max_files]):
        print(f"\n{'='*80}\nFile {i+1}: {file_path.name}")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "data" in data:
                print("Structure: Dictionary with 'data' key")
                other_keys = [k for k in data.keys() if k != "data"]
                if other_keys:
                    print(f"Other keys: {other_keys}")
                inner_data = data["data"]
                if isinstance(inner_data, list):
                    print(f"Data: List of {len(inner_data)} events")
                    events = inner_data
                elif isinstance(inner_data, dict):
                    print("Data: Single event dictionary")
                    events = [inner_data]
                else:
                    print(f"Data is of unexpected type: {type(inner_data)}")
                    continue
            elif isinstance(data, list):
                print(f"Structure: Direct list of {len(data)} events")
                events = data
            elif isinstance(data, dict):
                print(
                    f"Structure: Dictionary without 'data' key. Keys: {list(data.keys())}"
                )
                if "id" in data and "bookmakers" in data:
                    print("Appears to be a single event")
                    events = [data]
                else:
                    print("Not recognized as event data")
                    continue
            else:
                print(f"Unrecognized data structure: {type(data)}")
                continue

            if not events:
                print("No events found in the data")
                continue

            for event_idx, event in enumerate(events[:2]):
                print(f"\nExamining Event {event_idx + 1}:")
                if not isinstance(event, dict):
                    print(f"Event is not a dictionary: {type(event)}")
                    continue
                print(f"Event ID: {event.get('id', 'N/A')}")
                print(f"Sport: {event.get('sport_key', 'N/A')}")
                print(
                    f"Teams: {event.get('home_team', 'N/A')} vs {event.get('away_team', 'N/A')}"
                )

                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    print("No bookmakers found!")
                    continue
                print(f"Found {len(bookmakers)} bookmakers")
                book = bookmakers[0]
                print(f"First bookmaker: {book.get('key', 'unknown')}")
                markets = book.get("markets", [])
                if not markets:
                    print("No markets found!")
                    continue
                print(f"Found {len(markets)} markets")
                h2h_markets = [m for m in markets if m.get("key") == "h2h"]
                if not h2h_markets:
                    print("No h2h markets found!")
                    continue
                print(f"Found {len(h2h_markets)} h2h markets")
                market = h2h_markets[0]
                outcomes = market.get("outcomes", [])
                if not outcomes:
                    print("No outcomes found!")
                    continue
                print(f"H2h market has {len(outcomes)} outcomes:")
                for outcome_idx, outcome in enumerate(outcomes):
                    print(f"\n  Outcome {outcome_idx + 1}:")
                    print(f"    Name: {outcome.get('name', 'N/A')}")
                    print(f"    Price: {outcome.get('price', 'N/A')}")
                    print(f"    Result: {outcome.get('result', 'N/A')}")
                    print(f"    All fields: {list(outcome.keys())}")

                print("\nFull event structure:")
                try:
                    print(json.dumps(event, indent=2, default=str)[:500] + "...")
                except Exception:  # pragma: no cover - fallback pretty print
                    pprint.pprint(event)
        except Exception as e:  # pragma: no cover - diagnostic output only
            msg = f"Error examining file {file_path}: {e}"
            print(msg)
            logging.warning(msg)

    print("\nCache examination complete.")


def train_from_cache(cache_dir=CACHE_DIR, model_out=H2H_MODEL_PATH, verbose=True):
    """Attempt to build a training dataset from cached API data"""
    if verbose:
        print(f"Looking for cached data in {cache_dir}")

    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        msg = f"No cache files found in {cache_dir}"
        print(msg)
        logging.warning(msg)
        return False

    if verbose:
        print(f"Found {len(cache_files)} cache files")

    rows = []
    processed_files = 0

    for cache_file in cache_files:
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            # Handle the structure where data is in a 'data' field
            if isinstance(cached_data, dict) and 'data' in cached_data:
                data = cached_data['data']
            else:
                data = cached_data

            # Process data, which could be a list of events
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    # Process bookmakers
                    for book in item.get("bookmakers", []):
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

                            # Check if results are available
                            result1 = outcomes[0].get("result")
                            result2 = outcomes[1].get("result")

                            if None in (team1, team2, price1, price2):
                                continue

                            # If we have results, use them for training
                            if result1 is not None and result2 is not None:
                                label = 1 if result1 == "win" else 0
                                rows.append({
                                    "team1": team1,
                                    "team2": team2,
                                    "price1": price1,
                                    "price2": price2,
                                    "implied_prob": american_odds_to_prob(price1),
                                    "team1_win": label,
                                })
                                if verbose:
                                    print(f"Found h2h result: {team1}({price1}) vs {team2}({price2}) result: {result1}")

            processed_files += 1
            if verbose and processed_files % 100 == 0:
                print(f"Processed {processed_files} cache files...")

        except Exception as e:
            if verbose:
                msg = f"Error processing cache file {cache_file}: {e}"
                print(msg)
                logging.warning(msg)
            continue

    if not rows:
        msg = "No valid training data found in cache files"
        print(msg)
        logging.warning(msg)
        return False

    df = pd.DataFrame(rows)

    if verbose:
        print(f"Successfully created dataset with {len(df)} rows")
        print(df.head())

    X = df[["price1", "price2"]]
    y = df["team1_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize inputs then train logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probas)
    brier = brier_score_loss(y_test, probas)
    print(f"Model validation AUC: {auc:.3f}, Brier score: {brier:.3f}")

    residuals_df = pd.DataFrame({
        "true_label": y_test.reset_index(drop=True),
        "model_prob": probas,
    })
    residuals_df["residual"] = residuals_df["true_label"] - residuals_df["model_prob"]
    residuals_path = model_path.with_suffix(".residuals.csv")
    residuals_df.to_csv(residuals_path, index=False)
    print(f"Residuals saved to {residuals_path}")

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model successfully saved to {model_path}")
    return True


def train_simple_model(model_out=H2H_MODEL_PATH):
    """Create a simple model based on implied probability conversion"""
    # Using SimpleOddsModel defined at module level
    model = SimpleOddsModel()
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Simple odds conversion model saved to {model_path}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train h2h model from cache or API"
    )
    parser.add_argument(
        "--use-cache", action="store_true", help="Try to use cached data first"
    )
    parser.add_argument(
        "--examine-cache", action="store_true", help="Examine cache files to debug"
    )
    parser.add_argument(
        "--simple", action="store_true", help="Create a simple odds conversion model"
    )
    parser.add_argument(
        "--model-out", default=str(H2H_MODEL_PATH), help="Output model path"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()

    if args.examine_cache:
        examine_cache_files(max_files=10)
        sys.exit(0)

    success = False

    if args.use_cache:
        success = train_from_cache(model_out=args.model_out, verbose=args.verbose)

    if not success or args.simple:
        print("Creating a simple odds conversion model...")
        train_simple_model(model_out=args.model_out)
