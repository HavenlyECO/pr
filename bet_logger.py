# Utilities for logging bet recommendations and outcomes

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

if os.name == "nt":  # pragma: no cover - Windows only
    import msvcrt
else:  # pragma: no cover - POSIX
    import fcntl

from bankroll import calculate_bet_size
from ml import american_odds_to_payout, american_odds_to_prob


def _load_logs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _lock_file(f: "TextIO") -> None:
    """Acquire an exclusive lock on ``f``."""
    if os.name == "nt":  # pragma: no cover - Windows only
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    else:  # pragma: no cover - POSIX
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f: "TextIO") -> None:
    """Release the lock on ``f``."""
    if os.name == "nt":  # pragma: no cover - Windows only
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:  # pragma: no cover - POSIX
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _write_logs(path: Path, logs: list[dict]) -> None:
    """Write ``logs`` to ``path`` using an exclusive file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        _lock_file(f)
        try:
            f.seek(0)
            f.truncate()
            for entry in logs:
                json.dump(entry, f)
                f.write("\n")
        finally:
            _unlock_file(f)


def log_bets(
    projections: list[dict],
    *,
    threshold: float,
    bankroll: float | None = None,
    kelly_fraction: float = 1.0,
    log_file: str = "bet_log.jsonl",
) -> None:
    """Append qualifying bet recommendations to ``log_file``.

    Each entry records the team, odds, predicted probability, implied probability,
    edge and optional stake. Outcome fields remain ``null`` until updated
    later via :func:`update_bet_result`.
    """
    path = Path(log_file)
    logs = _load_logs(path)

    for row in projections:
        if row.get("risk_block_flag"):
            continue
        edge = row.get("weighted_edge", row.get("edge"))
        prob = row.get("projected_team1_win_probability")
        implied = row.get("implied_team1_win_probability")
        if edge is None or prob is None or implied is None or edge <= threshold:
            continue
        team = row.get("team1", "")
        odds = row.get("price1")
        bookmaker = row.get("bookmaker", "")
        event_id = row.get("event_id")
        timestamp = datetime.utcnow().isoformat()
        entry: dict = {
            "timestamp": timestamp,
            "event_id": event_id,
            "team": team,
            "odds": odds,
            "predicted_prob": prob,
            "implied_prob": implied,
            "edge": row.get("edge"),
            "weighted_edge": edge,
            "market_disagreement_score": row.get("market_disagreement_score"),
            "market_maker_mirror_score": row.get("market_maker_mirror_score"),
            "stale_line_flag": row.get("stale_line_flag", False),
            "bookmaker": bookmaker,
            "stake": None,
            "outcome": None,
            "payout": None,
            "roi": None,
        }
        if bankroll is not None:
            stake = calculate_bet_size(bankroll, prob, odds, fraction=kelly_fraction)
            entry["stake"] = round(stake, 2)
        logs.append(entry)

    if logs:
        _write_logs(path, logs)


def update_bet_result(
    event_id: str,
    team: str,
    result: Literal["win", "loss"],
    *,
    closing_odds: float | None = None,
    closing_implied_prob: float | None = None,
    log_file: str = "bet_log.jsonl",
) -> None:
    """Update the log entry for ``event_id`` and ``team`` with ``result``.

    When ``closing_odds`` or ``closing_implied_prob`` are supplied the function
    records the closing line's implied probability and a ``deviation_score``
    showing how far the model prediction deviated from the market at close.
    """
    path = Path(log_file)
    logs = _load_logs(path)
    updated = False

    for entry in logs:
        if entry.get("event_id") == event_id and entry.get("team") == team and entry.get("outcome") is None:
            stake = entry.get("stake") or 0.0
            odds = entry.get("odds")
            if result == "win":
                profit = stake * american_odds_to_payout(odds)
                payout = profit
            else:
                profit = -stake
                payout = -stake
            roi = 0.0
            if stake:
                roi = profit / stake

            if closing_implied_prob is None and closing_odds is not None:
                closing_implied_prob = american_odds_to_prob(closing_odds)
            if closing_implied_prob is not None:
                entry["closing_implied_prob"] = round(closing_implied_prob, 3)
                predicted = entry.get("predicted_prob")
                if predicted is not None:
                    deviation = predicted - closing_implied_prob
                    entry["deviation_score"] = round(deviation, 3)

            entry["outcome"] = result
            entry["payout"] = round(payout, 2)
            entry["roi"] = round(roi, 3)
            updated = True
            break

    if updated:
        _write_logs(path, logs)

