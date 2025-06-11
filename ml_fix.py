#!/usr/bin/env python3
from datetime import datetime
import os


def update_extract_advanced_ml_features():
    """Update the ml.py file with proper feature construction."""

    new_function = '''
def extract_advanced_ml_features(
    model_path: str,
    *,
    price1: float,
    price2: float,
    team1: str | None = None,
    team2: str | None = None,
) -> dict:
    """Return additional metrics from a moneyline or dual-head model with proper feature construction."""

    # Current time for datetime features
    now = datetime.utcnow()

    # Create complete features dict with all required columns
    features = {
        "price1": price1,
        "price2": price2,
        "pregame_price": price1,
        "pregame_line": price1,
        "event_id": "prediction",
        "commence_time": now.isoformat(),
        "bookmaker": "prediction",
        "date": now.date().isoformat(),
        "home_team": team1 or "Team1",
        "away_team": team2 or "Team2",
        "home_score": 0,
        "visiting_score": 0,
        "day_night": "D",
        "attendance": 0,
        "implied_prob": american_odds_to_prob(price1),
        "game_day": now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
    }

    try:
        prob = predict_moneyline_probability(model_path, features)
    except Exception as e:
        print(f"Failed to predict with advanced model: {e}")
        return {}

    implied = american_odds_to_prob(price1)
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
'''

    ml_path = "ml.py"
    with open(ml_path, 'r') as f:
        content = f.read()

    start_marker = "def extract_advanced_ml_features"
    end_marker = "def extract_market_signals"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Could not find extract_advanced_ml_features function")
        return False

    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print("Could not find end of function")
        return False

    new_content = content[:start_idx] + new_function + content[end_idx:]

    os.rename(ml_path, f"{ml_path}.bak")

    with open(ml_path, 'w') as f:
        f.write(new_content)

    print(f"Updated {ml_path} with fixed extract_advanced_ml_features function")
    print(f"Original file backed up as {ml_path}.bak")
    return True


if __name__ == "__main__":
    update_extract_advanced_ml_features()
