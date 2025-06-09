# Utilities for logging bet recommendations and outcomes

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from bankroll import calculate_bet_size
from ml import american_odds_to_payout


def _load_logs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_logs(path: Path, logs: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in logs:
            json.dump(entry, f)
            f.write("\n")


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
        edge = row.get("edge")
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
            "edge": edge,
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
    log_file: str = "bet_log.jsonl",
) -> None:
    """Update the log entry for ``event_id`` and ``team`` with ``result``."""
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
            entry["outcome"] = result
            entry["payout"] = round(payout, 2)
            entry["roi"] = round(roi, 3)
            updated = True
            break

    if updated:
        _write_logs(path, logs)

