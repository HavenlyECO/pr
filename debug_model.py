#!/usr/bin/env python3
import pickle
import pandas as pd
from pathlib import Path


def inspect_dual_head_model():
    """Inspect the model to understand what features it expects."""
    model_path = Path('h2h_data/h2h_classifier.pkl')

    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Determine model structure
    if isinstance(model_data, tuple):
        print("Model is a tuple of (model, columns)")
        model, columns = model_data
        print(f"Required columns ({len(columns)}):")
        for i, col in enumerate(columns):
            print(f"  {i+1}. {col}")
    else:
        print("Model is a direct model object")
        model = model_data

        if hasattr(model, 'pregame_cols'):
            print("Model has pregame_cols attribute:")
            for col in model.pregame_cols:
                print(f"  - {col}")

        if hasattr(model, 'live_cols'):
            print("Model has live_cols attribute:")
            for col in model.live_cols:
                print(f"  - {col}")

    # Print model type
    print(f"Model type: {type(model)}")

    # Test with sample data
    sample = {
        'price1': 100,
        'price2': -110,
        # Add all the columns that we've seen in warnings
        'pregame_price': 100,
        'pregame_line': 100,
        'event_id': 'test_event',
        'commence_time': '2025-06-11T07:53:30',
        'bookmaker': 'test_book',
        'date': '2025-06-11',
        'home_team': 'Team1',
        'away_team': 'Team2',
        'home_score': 0,
        'visiting_score': 0,
        'day_night': 'D',
        'attendance': 0,
        'implied_prob': 0.5,
        'game_day': 3,
        'is_weekend': 0,
    }

    # Test prediction with sample data
    try:
        from ml import predict_moneyline_probability
        prob = predict_moneyline_probability(str(model_path), sample)
        print(f"Prediction successful: {prob:.4f}")
    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    inspect_dual_head_model()
