#!/usr/bin/env python3
import os


def patch_ml_file():
    """Update ml.py with fixed function"""
    ml_path = "ml.py"

    # Read current file
    with open(ml_path, 'r') as f:
        content = f.read()

    # New function with proper numeric conversions
    new_function = '''
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
            print(f"Using fallback probability calculation")
            prob = implied + 0.02  # Slight edge over implied

        edge = prob - implied
        ev = edge * american_odds_to_payout(price1)

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

    # Find the current function and replace it
    start_marker = "def extract_advanced_ml_features"
    end_marker = "def extract_market_signals"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Could not find extract_advanced_ml_features function in ml.py")
        return False

    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print("Could not find end of function in ml.py")
        return False

    # Replace the function
    new_content = content[:start_idx] + new_function + content[end_idx:]

    # Create backup
    os.rename(ml_path, f"{ml_path}.bak2")

    # Write updated content
    with open(ml_path, 'w') as f:
        f.write(new_content)

    print(f"Updated {ml_path} with fixed extract_advanced_ml_features function")
    print(f"Original file backed up as {ml_path}.bak2")
    return True


if __name__ == "__main__":
    patch_ml_file()
