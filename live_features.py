from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable

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


class OffensivePressureTracker:
    """Track recent hits, errors and scoring chances."""

    def __init__(self) -> None:
        self._hits_by_inning: Dict[int, int] = {}
        self._lob_by_inning: Dict[int, int] = {}
        self._errors_by_inning: Dict[int, int] = {}
        self._risp_by_inning: Dict[int, float] = {}

    def update(
        self,
        inning: int,
        home_hits: int,
        away_hits: int,
        home_lob: int,
        away_lob: int,
        *,
        home_errors: int | None = None,
        away_errors: int | None = None,
        home_risp: float | None = None,
        away_risp: float | None = None,
    ) -> None:
        """Record offensive pressure metrics for a completed inning."""

        self._hits_by_inning[inning] = home_hits - away_hits
        self._lob_by_inning[inning] = home_lob - away_lob
        if None not in (home_errors, away_errors):
            self._errors_by_inning[inning] = home_errors - away_errors
        if None not in (home_risp, away_risp):
            self._risp_by_inning[inning] = home_risp - away_risp

    def _sum_last(self, data: Dict[int, float], n: int) -> float:
        """Sum values from the last ``n`` sequential innings.

        ``data`` is keyed by inning number. If innings are skipped or
        misnumbered, simply taking the largest ``n`` keys could lead to
        surprising results. Instead, determine the most recent inning and
        sum over the preceding ``n`` innings (using ``0`` for missing
        innings).
        """

        if not data or n <= 0:
            return 0.0

        last_inning = max(data)
        start = max(1, last_inning - n + 1)
        innings_range = range(start, last_inning + 1)
        return sum(data.get(i, 0) for i in innings_range)

    def feature_dict(self, n: int = 2) -> Dict[str, float]:
        """Return mapping with momentum features for the last ``n`` innings."""

        feats: Dict[str, float] = {
            f"hits_last_{n}_innings": self._sum_last(self._hits_by_inning, n),
            f"LOB_last_{n}": self._sum_last(self._lob_by_inning, n),
        }
        if self._errors_by_inning:
            feats[f"errors_last_{n}"] = self._sum_last(self._errors_by_inning, n)
        if self._risp_by_inning:
            feats[f"RISP_last_{n}"] = self._sum_last(self._risp_by_inning, n)
        return feats


class WinProbabilitySwingTracker:
    """Track win probability swings after each inning."""

    def __init__(self, model_path: str, base_features: Dict[str, float] | None = None) -> None:
        self.model_path = model_path
        self.base_features = base_features or {}
        self._diff_tracker = InningDifferentialTracker()
        self._pressure_tracker = OffensivePressureTracker()
        self._prob_by_inning: Dict[int, float] = {}
        self._delta_by_inning: Dict[int, float] = {}

    def update(
        self,
        inning: int,
        home_runs: int,
        away_runs: int,
        *,
        home_hits: int | None = None,
        away_hits: int | None = None,
        home_lob: int | None = None,
        away_lob: int | None = None,
        home_errors: int | None = None,
        away_errors: int | None = None,
        home_risp: float | None = None,
        away_risp: float | None = None,
    ) -> None:
        """Update win probability using latest score."""

        from ml import predict_moneyline_probability

        self._diff_tracker.update(inning, home_runs, away_runs)
        features = {**self.base_features, **self._diff_tracker.feature_dict()}
        if None not in (home_hits, away_hits, home_lob, away_lob):
            self._pressure_tracker.update(
                inning,
                home_hits,
                away_hits,
                home_lob,
                away_lob,
                home_errors=home_errors,
                away_errors=away_errors,
                home_risp=home_risp,
                away_risp=away_risp,
            )
            features.update(self._pressure_tracker.feature_dict())
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
    offensive_stats: Iterable[
        tuple[
            int,
            int,
            int,
            int,
            int | None,
            int | None,
            float | None,
            float | None,
        ]
    ] | None = None,
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
    offensive_stats:
        Iterable of per-inning stats. Each tuple should contain
        ``(home_hits, away_hits, home_lob, away_lob, home_errors, away_errors,
        home_risp, away_risp)``. The error and RISP values can be ``None`` if
        unavailable.

    Returns
    -------
    list of dict
        Each entry contains ``inning``, ``timestamp`` and ``probability`` fields
        that can be plotted over time.
    """

    from ml import predict_moneyline_probability

    tracker = InningDifferentialTracker()
    pressure = OffensivePressureTracker() if offensive_stats is not None else None
    curve: list[dict] = []
    extras = base_features or {}
    stats_iter = iter(offensive_stats) if offensive_stats is not None else None

    for inning, home, away in inning_scores:
        tracker.update(inning, home, away)
        features = {**extras, **tracker.feature_dict()}
        if pressure is not None and stats_iter is not None:
            stats = next(stats_iter)
            h_hits, a_hits, h_lob, a_lob = stats[:4]
            extras_stats = {
                "home_errors": None,
                "away_errors": None,
                "home_risp": None,
                "away_risp": None,
            }
            if len(stats) >= 6:
                extras_stats["home_errors"] = stats[4]
                extras_stats["away_errors"] = stats[5]
            if len(stats) == 8:
                extras_stats["home_risp"] = stats[6]
                extras_stats["away_risp"] = stats[7]
            pressure.update(
                inning,
                h_hits,
                a_hits,
                h_lob,
                a_lob,
                home_errors=extras_stats["home_errors"],
                away_errors=extras_stats["away_errors"],
                home_risp=extras_stats["home_risp"],
                away_risp=extras_stats["away_risp"],
            )
            features.update(pressure.feature_dict())
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
