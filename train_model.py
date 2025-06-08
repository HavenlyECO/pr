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
    """Examine what's in the cache files to help debug training issues"""
    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        print(f"No cache files found in {cache_dir}")
        return

    print(f"Examining first {min(max_files, len(cache_files))} cache files:")

    for i, cache_file in enumerate(cache_files[:max_files]):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            print(f"\nFile {i+1}: {cache_file.name}")

            if isinstance(data, dict):
                print(f"  Contains a dictionary with keys: {list(data.keys())}")

                # Look inside the 'data' field which may contain the actual API response
                if 'data' in data:
                    inner_data = data['data']

                    if isinstance(inner_data, list):
                        print(f"  Data field contains a list of {len(inner_data)} items")

                        if inner_data:
                            for idx, item in enumerate(inner_data[:1]):  # Look at first item
                                print(f"  Item {idx} type: {type(item)}")

                                if isinstance(item, dict):
                                    print(f"  Item {idx} keys: {list(item.keys())}")

                                    # Check for bookmakers
                                    if 'bookmakers' in item:
                                        print(f"  Found {len(item['bookmakers'])} bookmakers")

                                        for b_idx, book in enumerate(item['bookmakers'][:1]):  # First bookmaker
                                            print(f"  Bookmaker {b_idx}: {book.get('key', 'unknown')}")

                                            # Check for markets
                                            if 'markets' in book:
                                                print(f"  Found {len(book['markets'])} markets")

                                                for m_idx, market in enumerate(book['markets'][:1]):  # First market
                                                    print(f"  Market {m_idx} key: {market.get('key', 'unknown')}")

                                                    # Check for h2h market
                                                    if market.get('key') == 'h2h' and 'outcomes' in market:
                                                        print(f"  Found h2h market with {len(market['outcomes'])} outcomes")

                                                        # Check for results field
                                                        has_results = any('result' in outcome for outcome in market['outcomes'])
                                                        print(f"  Has results field: {has_results}")

                                                        for o_idx, outcome in enumerate(market['outcomes']):
                                                            print(f"  Outcome {o_idx} keys: {list(outcome.keys())}")
                    elif isinstance(inner_data, dict):
                        print(f"  Data field contains a dictionary with keys: {list(inner_data.keys())}")
            elif isinstance(data, list):
                print(f"  Contains a list of {len(data)} items")

                if data and isinstance(data[0], dict):
                    print(f"  First item keys: {list(data[0].keys())}")

        except Exception as e:
            print(f"  Error examining file {cache_file}: {e}")

    print("\nCache examination complete.")


def train_from_cache(cache_dir=CACHE_DIR, model_out=H2H_MODEL_PATH, verbose=True):
    """Attempt to build a training dataset from cached API data"""
    if verbose:
        print(f"Looking for cached data in {cache_dir}")

    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        print(f"No cache files found in {cache_dir}")
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
                print(f"Error processing cache file {cache_file}: {e}")
            continue

    if not rows:
        print("No valid training data found in cache files")
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
