import pickle
from pathlib import Path
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / 'h2h_data' / 'h2h_classifier.pkl'


def create_simplified_h2h_model(model_path: Path = MODEL_PATH) -> None:
    """Wrap the saved h2h model to only require price columns."""
    with open(model_path, 'rb') as f:
        dual_head_model = pickle.load(f)

    class SimpleH2HModel:
        def __init__(self, dual_head_model):
            self.dual_head_model = dual_head_model

        def predict_proba(self, X: pd.DataFrame):
            price1 = X['price1'].values[0] if 'price1' in X.columns else X.iloc[0, 0]
            price2 = X['price2'].values[0] if 'price2' in X.columns else X.iloc[0, 1]

            features = {
                'price1': price1,
                'price2': price2,
                'pregame_price': price1,
                'pregame_line': price1,
                'implied_prob': 100 / (price1 + 100) if price1 > 0 else abs(price1) / (abs(price1) + 100),
                'event_id': 'prediction',
                'commence_time': None,
                'bookmaker': 'prediction',
                'date': None,
                'home_team': 'Team1',
                'away_team': 'Team2',
                'home_score': 0,
                'visiting_score': 0,
                'day_night': 'D',
                'attendance': 0,
                'team1': 'Team1',
                'team2': 'Team2',
                'game_day': 0,
                'is_weekend': 0,
            }

            if not hasattr(self.dual_head_model, 'pregame_cols'):
                return self.dual_head_model.predict_proba(X)

            df = pd.DataFrame([features])
            return self.dual_head_model.predict_proba(df)

    with open(model_path, 'wb') as f:
        pickle.dump((SimpleH2HModel(dual_head_model), ['price1', 'price2']), f)

    print(f"Created and saved simplified h2h model wrapper to {model_path}")


if __name__ == '__main__':
    create_simplified_h2h_model()
