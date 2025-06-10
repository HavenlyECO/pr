#!/usr/bin/env python3
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from ml import CACHE_DIR
from scores import fetch_scores, append_scores_history


def update_cache_with_scores(sport_key="baseball_mlb", days_back=3, verbose=True):
    """Update cache files with results from the scores API"""
    scores = fetch_scores(sport_key, days_from=days_back)
    if not scores:
        if verbose:
            print("No scores fetched from API")
        return 0

    # Create a mapping of game IDs to results
    results = {}
    for game in scores:
        game_id = game.get("id")
        if not game_id:
            continue

        # Extract completed games with scores
        scores_data = game.get("scores")
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        home_score = None
        away_score = None

        if isinstance(scores_data, dict):
            home_score = scores_data.get("home", {}).get("score")
            away_score = scores_data.get("away", {}).get("score")
        elif isinstance(scores_data, list):
            # Loop through the scores list to find home and away scores
            for score_item in scores_data:
                team_name = score_item.get("name")
                score = score_item.get("score")
                if team_name == home_team:
                    home_score = score
                elif team_name == away_team:
                    away_score = score

        completed = game.get("completed", False)

        # Only process completed games with valid scores
        if not completed or home_score is None or away_score is None:
            continue

        # Determine winners
        home_win = home_score > away_score
        away_win = away_score > home_score

        results[game_id] = {
            "home_team": home_team,
            "away_team": away_team,
            "home_win": home_win,
            "away_win": away_win,
        }

    if verbose:
        print(f"Found results for {len(results)} completed games")

    # Now update cache files with these results
    updated_files = 0
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    for cache_path in cache_files:
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            modified = False
            data = cached.get("data") if isinstance(cached, dict) and "data" in cached else cached

            # Handle different data structures
            if isinstance(data, dict) and "id" in data:  # Single event
                game_id = data.get("id")
                if game_id in results:
                    modified = update_outcomes(data, results[game_id])

            elif isinstance(data, list):  # Multiple events
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    game_id = item.get("id")
                    if game_id in results:
                        if update_outcomes(item, results[game_id]):
                            modified = True

            if modified:
                # Save the updated cache file
                with open(cache_path, "wb") as f:
                    pickle.dump(cached, f)
                updated_files += 1

        except Exception as e:
            if verbose:
                print(f"Error processing {cache_path}: {e}")

    if verbose:
        print(f"Updated {updated_files} cache files with results")

    # Also save scores to history file for reference
    append_scores_history(scores)

    return updated_files


def update_outcomes(game_data, result):
    """Update the outcomes in a game with win/loss results"""
    if not isinstance(game_data, dict) or "bookmakers" not in game_data:
        return False

    modified = False
    for book in game_data.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue

            outcomes = market.get("outcomes", [])
            if len(outcomes) != 2:
                continue

            # Process both outcomes in the market
            for outcome in outcomes:
                team_name = outcome.get("name")
                if not team_name:
                    continue

                # Match team name to result
                if team_name == result["home_team"]:
                    outcome["result"] = "win" if result["home_win"] else "loss"
                    modified = True
                elif team_name == result["away_team"]:
                    outcome["result"] = "win" if result["away_win"] else "loss"
                    modified = True

    return modified


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update cache with recent game results")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--days-back", type=int, default=3,
                        help="How many days of history to fetch (max 3)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously, updating daily")
    parser.add_argument("--interval-hours", type=int, default=6,
                        help="Hours between updates when running continuously")

    args = parser.parse_args()

    if args.continuous:
        import time
        print(f"Running continuous updates every {args.interval_hours} hours")
        while True:
            update_cache_with_scores(args.sport, args.days_back, args.verbose)
            # Sleep until next update
            time.sleep(args.interval_hours * 3600)
    else:
        update_cache_with_scores(args.sport, args.days_back, args.verbose)
