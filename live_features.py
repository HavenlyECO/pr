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
