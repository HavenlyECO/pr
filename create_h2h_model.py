import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def create_new_h2h_model() -> bool:
    """Create a simple h2h model from CSV data."""
    model_path = Path(__file__).resolve().parent / 'h2h_data' / 'h2h_classifier.pkl'

    # Ensure the directory exists
    model_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"Creating new h2h model at {model_path}")

    try:
        df = pd.read_csv('integrated_training_data.csv')
        print(f"Loaded training data with {len(df)} rows")

        X = df[['price1', 'price2']]
        y = df['home_team_win']

        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        accuracy = pipeline.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")

        with open(model_path, 'wb') as f:
            pickle.dump((pipeline, list(X.columns)), f)

        print(f"Successfully saved new h2h model to {model_path}")
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        return False


if __name__ == '__main__':
    create_new_h2h_model()
