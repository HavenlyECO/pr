from __future__ import annotations

from datetime import datetime
from typing import Dict

class InningDifferentialTracker:
    """Track inning-by-inning run differential with timestamps."""

    def __init__(self) -> None:
        # Mapping of inning number -> differential (home minus away)
        self._diff_by_inning: Dict[int, int] = {}
        # Timestamp of when each inning update was recorded (UTC)
        self._timestamp_by_inning: Dict[int, datetime] = {}

    def update(self, inning: int, home_runs: int, away_runs: int) -> None:
        """Update the differential for a completed inning."""
        diff = home_runs - away_runs
        self._diff_by_inning[inning] = diff
        self._timestamp_by_inning[inning] = datetime.utcnow()

    def feature_dict(self) -> Dict[str, int]:
        """Return feature mapping like {'live_inning_1_diff': 2, ...}."""
        return {
            f"live_inning_{inning}_diff": diff
            for inning, diff in sorted(self._diff_by_inning.items())
        }

    def cumulative_diff(self) -> int:
        """Return the cumulative run differential so far."""
        return sum(self._diff_by_inning.values())


class WinProbabilitySwingTracker:
    """Track win probability swings after each inning."""

    def __init__(self, model_path: str, base_features: Dict[str, float] | None = None) -> None:
        self.model_path = model_path
        self.base_features = base_features or {}
        self._diff_tracker = InningDifferentialTracker()
        self._prob_by_inning: Dict[int, float] = {}
        self._delta_by_inning: Dict[int, float] = {}

    def update(self, inning: int, home_runs: int, away_runs: int) -> None:
        """Update win probability using latest score."""

        from ml import predict_moneyline_probability

        self._diff_tracker.update(inning, home_runs, away_runs)
        features = {**self.base_features, **self._diff_tracker.feature_dict()}
        prob = predict_moneyline_probability(self.model_path, features)
        prev_inning = max((i for i in self._prob_by_inning if i < inning), default=None)
        if prev_inning is None:
            delta = 0.0
        else:
            delta = prob - self._prob_by_inning[prev_inning]
        self._prob_by_inning[inning] = prob
        self._delta_by_inning[inning] = delta

    def swing_features(self) -> Dict[str, float]:
        """Return mapping like {'win_prob_delta_inning_1': -0.05, ...}."""

        return {
            f"win_prob_delta_inning_{inning}": delta
            for inning, delta in sorted(self._delta_by_inning.items())
        }

    def curve(self) -> list[dict]:
        """Return win probability curve with deltas."""

        return [
            {
                "inning": inning,
                "timestamp": self._diff_tracker._timestamp_by_inning[inning],
                "probability": prob,
                "delta": self._delta_by_inning.get(inning, 0.0),
            }
            for inning, prob in sorted(self._prob_by_inning.items())
        ]


def build_win_probability_curve(
    model_path: str,
    inning_scores: list[tuple[int, int, int]],
    base_features: Dict[str, int] | None = None,
) -> list[dict]:
    """Generate win probability after each inning.

    Parameters
    ----------
    model_path:
        Path to a trained moneyline classifier.
    inning_scores:
        Sequence of ``(inning, home_runs, away_runs)`` tuples representing the
        score at the end of each inning.
    base_features:
        Additional features to include in every prediction (e.g. pregame stats).

    Returns
    -------
    list of dict
        Each entry contains ``inning``, ``timestamp`` and ``probability`` fields
        that can be plotted over time.
    """

    from ml import predict_moneyline_probability

    tracker = InningDifferentialTracker()
    curve: list[dict] = []
    extras = base_features or {}

    for inning, home, away in inning_scores:
        tracker.update(inning, home, away)
        features = {**extras, **tracker.feature_dict()}
        prob = predict_moneyline_probability(model_path, features)
        timestamp = tracker._timestamp_by_inning[inning]
        curve.append(
            {
                "inning": inning,
                "timestamp": timestamp,
                "probability": prob,
            }
        )

    return curve
