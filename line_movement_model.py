"""Utilities for training and evaluating line movement models."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def load_and_engineer_features(csv_path: str) -> pd.DataFrame:
    """Return a dataframe with engineered features for line movement modeling."""
    df = pd.read_csv(csv_path)

    def american_odds_to_prob(odds: float) -> float:
        return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

    df["implied_prob_open"] = df["opening_odds"].apply(american_odds_to_prob)
    df["implied_prob_close"] = df["closing_odds"].apply(american_odds_to_prob)

    df["line_shift"] = df["closing_odds"] - df["opening_odds"]
    df["direction"] = np.where(df["line_shift"] > 0, 1, np.where(df["line_shift"] < 0, -1, 0))

    tick_bins = [-np.inf, -10, -1, 1, 10, np.inf]
    tick_labels = ["big_fav", "small_fav", "static", "small_dog", "big_dog"]
    df["tick_bin"] = pd.cut(df["line_shift"], bins=tick_bins, labels=tick_labels)

    df["interval_hours"] = (
        pd.to_datetime(df["timestamp_close"]) - pd.to_datetime(df["timestamp_open"])
    ).dt.total_seconds() / 3600.0

    df = df.dropna(
        subset=[
            "opening_odds",
            "closing_odds",
            "volatility",
            "implied_prob_open",
            "implied_prob_close",
            "interval_hours",
        ]
    )
    return df


def train_regression_model(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
):
    """Train and return a regression pipeline for line shift prediction."""
    X = df[feature_cols]
    y = df[target_col]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
    print(f"Regression MAE CV: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    pipeline.fit(X, y)
    return pipeline


def train_classification_model(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
):
    """Train and return a classification pipeline for line shift bins."""
    X = df[feature_cols]
    y = df[target_col]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    print(f"Classification Accuracy CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    pipeline.fit(X, y)
    return pipeline


def evaluate_regression(model, X_test, y_test) -> None:
    """Print MAE and RMSE on the test split."""
    y_pred = model.predict(X_test)
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))


def evaluate_classification(model, X_test, y_test) -> None:
    """Print accuracy metrics on the test split."""
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = load_and_engineer_features("line_movement_data.csv")
    feature_cols = [
        "opening_odds",
        "implied_prob_open",
        "volatility",
        "interval_hours",
    ]

    reg_model = train_regression_model(df, feature_cols, "line_shift")
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["line_shift"], test_size=0.2, random_state=42
    )
    evaluate_regression(reg_model, X_test, y_test)

    cls_model = train_classification_model(df, feature_cols, "tick_bin")
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df["tick_bin"],
        test_size=0.2,
        random_state=42,
        stratify=df["tick_bin"],
    )
    evaluate_classification(cls_model, X_test, y_test)
