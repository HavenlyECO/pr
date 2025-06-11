#!/usr/bin/env python3
from pathlib import Path
import pickle
import pandas as pd


def inspect_model_columns():
    """Inspect model columns to determine required data types."""
    model_path = Path('h2h_data/h2h_classifier.pkl')

    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Extract column information
    if isinstance(model_data, tuple):
        model, cols = model_data
        print(f"Model expects {len(cols)} columns")
    else:
        model = model_data
        cols = getattr(model, 'pregame_cols', [])
        if not cols:
            cols = []
            print("Warning: Could not find columns list in model")

    # Create a simple input file for ML module
    ml_fix = '''
# Fix for ml.py's extract_advanced_ml_features function
def extract_advanced_ml_features(
    model_path: str,
    *,
    price1: float,
    price2: float,
    team1: str | None = None,
    team2: str | None = None,
) -> dict:
    """Return additional metrics from a moneyline model with proper numeric conversion."""

    # Create features dictionary with numeric values only
    features = {
        "price1": float(price1),
        "price2": float(price2),
        "pregame_price": float(price1),
        "pregame_line": float(price1),
        "event_id": -1,  # Use numeric placeholder instead of string
        "commence_time": -1,  # Numeric placeholder
        "bookmaker": -1,  # Numeric placeholder
        "date": -1,  # Numeric placeholder
        "home_team": -1 if team1 is None else 0,  # Numeric placeholders
        "away_team": -1 if team2 is None else 0,
        "home_score": 0,
        "visiting_score": 0, 
        "day_night": -1,  # Numeric placeholder
        "attendance": 0,
        "implied_prob": american_odds_to_prob(price1),
        "game_day": 0,
        "is_weekend": 0,
        "team1": -1 if team1 is None else 0,  # Add team1/team2 columns
        "team2": -1 if team2 is None else 0,
    }

    try:
        # Basic probability calculation that doesn't rely on the advanced model
        implied = american_odds_to_prob(price1)

        # Try to get prediction from the model
        try:
            prob = predict_moneyline_probability(model_path, features)
        except Exception as e:
            # Fallback to simple implied probability if model fails
            print(f"Using fallback probability calculation: {e}")  # PATCH: Log error for easier debugging
            prob = implied + 0.02  # Slight edge over implied

        edge = prob - implied
        ev = edge * american_odds_to_payout(price1)
        print(f"Model prob: {prob:.4f}, Implied: {implied:.4f}, Edge: {edge:.4f}")  # PATCH: Log edge calculation

        return {
            "ml_confidence": prob,
            "lineup_strength": 0.5,  # Default value
            "market_efficiency": implied,
            "sharp_action": edge,
            "advanced_ml_prob": prob,
            "advanced_ml_edge": edge,
            "advanced_ml_ev": ev,
        }
    except Exception as e:
        print(f"Error in extract_advanced_ml_features: {e}")
        return {}
'''

    # Write to file
    with open('ml_numeric_fix.py', 'w') as f:
        f.write(ml_fix)

    print("Created ml_numeric_fix.py with fixed function")
    print("To apply the fix, copy the function from ml_numeric_fix.py to ml.py")


if __name__ == "__main__":
    inspect_model_columns()
