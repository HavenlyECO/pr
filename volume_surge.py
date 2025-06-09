"""
Utilities for detecting real-time betting volume spikes.

This module provides a ``VolumeSurgeDetector`` that monitors matched volume
from an exchange via a user-supplied callback and computes a
``volume_surge_score`` when the latest volume deviates sharply from recent
history.
"""

import time
import statistics
import logging
from collections import deque
from typing import Callable, Deque, Tuple

class VolumeSurgeDetector:
    """Track short-term volume spikes on betting exchanges."""

    def __init__(self, fetch_volume: Callable[[], float], *, window_seconds: int = 600, z_threshold: float = 2.0) -> None:
        """Create detector.

        Parameters
        ----------
        fetch_volume:
            Callable that returns the current matched volume for the market.
        window_seconds:
            How many seconds of history to keep for surge calculations.
        z_threshold:
            Number of standard deviations above the mean required to flag a surge.
        """
        self.fetch_volume = fetch_volume
        self.window_seconds = window_seconds
        self.z_threshold = z_threshold
        self.history: Deque[Tuple[float, float]] = deque()

    def _trim_history(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def update(self) -> float:
        """Fetch the latest volume and return the current surge score."""
        try:
            volume = self.fetch_volume()
        except Exception:  # pragma: no cover - passthrough for unexpected errors
            logging.exception("fetch_volume callback failed")
            raise
        self.history.append((time.time(), volume))
        self._trim_history()
        return self.volume_surge_score()

    def volume_surge_score(self) -> float:
        """Return a score from 0-1 indicating abnormal volume."""
        if len(self.history) < 2:
            return 0.0
        volumes = [v for _, v in self.history]
        avg = statistics.mean(volumes)
        if len(volumes) > 1:
            stdev = statistics.stdev(volumes)
        else:
            stdev = 0.0
        if stdev <= 0:
            return 0.0
        z = (volumes[-1] - avg) / stdev
        score = max(0.0, min(1.0, z / self.z_threshold))
        return score
