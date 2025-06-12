"""Utilities for bankroll management and bet sizing."""

from __future__ import annotations

from typing import Literal
from odds_utils import american_odds_to_payout


def kelly_bet_fraction(prob: float, odds: float) -> float:
    """Return the Kelly bet fraction for a win probability and American odds.

    Implements ``(b * p - q) / b`` where ``b`` is the profit on a $1 bet
    (decimal odds minus 1), ``p`` is the model probability and ``q`` is
    ``1 - p``.
    """

    b = american_odds_to_payout(odds)
    p = prob
    q = 1 - p
    fraction = (b * p - q) / b
    return max(fraction, 0.0)


def calculate_bet_size(
    bankroll: float,
    prob: float,
    odds: float,
    *,
    fraction: float = 1.0,
) -> float:
    """Return recommended stake using Kelly criterion.

    ``fraction`` scales the full Kelly stake (e.g. 0.5 for half-Kelly).
    """
    kelly_fraction = kelly_bet_fraction(prob, odds)
    stake = bankroll * fraction * kelly_fraction
    return round(stake, 2)


# Simple bankroll update helper

def update_bankroll(
    bankroll: float,
    stake: float,
    result: Literal["win", "loss", "push", "void"],
    odds: float,
) -> float:
    """Return bankroll after applying bet result.

    ``result`` may be ``"win"``, ``"loss"``, ``"push"`` or ``"void"``.
    ``"push"`` and ``"void"`` leave the bankroll unchanged. Any other
    value raises ``ValueError`` to prevent silent mistakes.
    """

    if result == "win":
        profit = stake * american_odds_to_payout(odds)
        new_bankroll = bankroll + profit
    elif result == "loss":
        new_bankroll = bankroll - stake
    elif result in ("push", "void"):
        new_bankroll = bankroll
    else:
        raise ValueError(f"Unknown result '{result}'")

    return round(new_bankroll, 2)
