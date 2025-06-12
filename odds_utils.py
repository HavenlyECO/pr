"""Utility functions for working with American odds."""

from __future__ import annotations


def american_odds_to_prob(odds: float) -> float:
    """Return the implied probability from American odds."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_odds_to_payout(odds: float, stake: float = 1.0) -> float:
    """Return the total return for a given stake at the specified odds."""
    if odds > 0:
        return stake * (odds / 100) + stake
    return stake * (100 / abs(odds)) + stake

