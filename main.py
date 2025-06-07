import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv is required. Install it with 'pip install python-dotenv'")
    sys.exit(1)

# Always load .env from the project root
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)
else:
    print(f"Warning: .env file not found at {DOTENV_PATH}")

API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    print("THE_ODDS_API_KEY environment variable is not set. Please set it in your .env file.")
    sys.exit(1)


def build_odds_url(
    sport_key: str,
    *,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    game_period_markets: str | None = None,
) -> str:
    """Return fully qualified odds API URL."""
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    if game_period_markets:
        params["gamePeriodMarkets"] = game_period_markets
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_odds(
    sport_key: str,
    *,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    game_period_markets: str | None = None,
):
    """Fetch upcoming odds for ``sport_key``."""
    url = build_odds_url(
        sport_key,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        game_period_markets=game_period_markets,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return []


def build_historical_odds_url(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    include_scores: bool = False,
) -> str:
    """Return fully qualified historical odds API URL."""
    base_url = f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "date": date,
    }
    if include_scores:
        params["include"] = "scores"
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_historical_odds(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
):
    """Fetch historical odds for a given sport on a specific date."""
    url = build_historical_odds_url(
        sport_key,
        date=date,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        include_scores=True,
    )
    print(f"[DEBUG] Requesting historical odds from: {url}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            print(f"[DEBUG] HTTP Status: {resp.status}")
            print(f"[DEBUG] HTTP Headers: {resp.headers}")
            raw = resp.read()
            print(f"[DEBUG] Raw API response: {raw!r}")
            data = json.loads(raw.decode())
            print(f"[DEBUG] Parsed API response: {data!r}")
            return data
    except urllib.error.HTTPError as e:
        print(f"[DEBUG] HTTPError: {e.code} {e.reason}")
        if hasattr(e, "headers"):
            print(f"[DEBUG] Error Headers: {e.headers}")
        try:
            error_body = e.read()
            print(f"[DEBUG] Error Body: {error_body!r}")
            error_data = json.loads(error_body.decode())
            print(f"[DEBUG] Parsed Error Body: {error_data!r}")
        except Exception as err:
            print(f"[DEBUG] Could not parse error body: {err}")
        message = error_body.decode() if 'error_body' in locals() else str(e)
        raise RuntimeError(f"Failed to fetch historical odds: {message}") from e
    except Exception as e:
        print(f"[DEBUG] Unexpected Exception: {e}")
        raise


def _format_header(idx: int, game: dict) -> str:
    home = game.get("home_team", "N/A")
    away = game.get("away_team", "N/A")
    time = game.get("commence_time", "")
    return f"{idx}. {home} vs {away} ({time})"


def format_moneyline(games: list[dict]) -> str:
    """Return a human readable moneyline summary."""
    lines: list[str] = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Head-to-Head (Moneyline):")
        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = [f"{o.get('name', '')} {o.get('price', '')}" for o in market.get("outcomes", [])]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")
    return "\n".join(lines)


def format_projected_ks_props(games: list[dict], model_path: str) -> str:
    """Return strikeout prop odds with ML projected over probability."""
    from ml import predict_pitcher_ks_over_probability

    lines: list[str] = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_strikeouts":
                    continue
                pitcher_lines: dict[tuple, dict] = {}
                for outcome in market.get("outcomes", []):
                    pitcher = outcome.get("name")
                    line = outcome.get("line")
                    desc = outcome.get("description", "").lower()
                    if pitcher is None or line is None:
                        continue
                    key = (pitcher, line)
                    if key not in pitcher_lines:
                        pitcher_lines[key] = {
                            "pitcher": pitcher,
                            "line": line,
                            "price_over": None,
                            "price_under": None,
                        }
                    if desc.startswith("over"):
                        pitcher_lines[key]["price_over"] = outcome.get("price")
                    elif desc.startswith("under"):
                        pitcher_lines[key]["price_under"] = outcome.get("price")
                for props in pitcher_lines.values():
                    if props["price_over"] is None or props["price_under"] is None:
                        continue
                    features = {
                        "line": props["line"],
                        "price_over": props["price_over"],
                        "price_under": props["price_under"],
                    }
                    prob = predict_pitcher_ks_over_probability(model_path, features)
                    lines.append(
                        f"   {bm_title} {props['pitcher']} O{props['line']} "
                        f"{prob:.3f} (O {props['price_over']} U {props['price_under']})"
                    )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and display odds")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "moneyline",
            "spreads",
            "totals",
            "outrights",
            "alternate_spreads",
            "alternate_totals",
            "team_totals",
            "alternate_team_totals",
            "player_props",
            "projected_ks_props",
            "projected_ks_ml_eval",
            "historical",
            "train_classifier",
            "predict_classifier",
            "continuous_train_classifier",
        ],
        default="moneyline",
        help="Type of odds to display",
    )
    parser.add_argument("--sport", default="baseball_mlb", help="Sport key, e.g. baseball_mlb")
    parser.add_argument(
        "--game-period-markets",
        default=None,
        help="Comma separated game period markets to include, e.g. first_half_totals",
    )
    parser.add_argument("--dataset", default=None, help="CSV file for training the moneyline classifier")
    parser.add_argument("--model-out", default="moneyline_classifier.pkl", help="Where to save the trained classifier")
    parser.add_argument("--model", default="moneyline_classifier.pkl", help="Path to a trained classifier for prediction")
    parser.add_argument("--features", default=None, help="JSON string of feature values for prediction")
    parser.add_argument("--date", default=None, help="Date for historical odds in YYYY-MM-DD format")
    parser.add_argument("--start-date", default=None, help="Start date for training data in YYYY-MM-DD format")
    parser.add_argument("--end-date", default=None, help="End date for training data in YYYY-MM-DD format")
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=24,
        help="Interval between training runs in hours for continuous training",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)
        print("Verbose logging enabled.")

    # ML related imports deferred until needed
    if args.command == "train_classifier":
        from ml import train_classifier, train_classifier_df, build_dataset_from_api

        if args.dataset:
            train_classifier(args.dataset, model_out=args.model_out)
        else:
            if not args.start_date or not args.end_date:
                parser.error("--dataset or --start-date/--end-date required for train_classifier")
            df = build_dataset_from_api(
                args.sport,
                args.start_date,
                args.end_date,
            )
            train_classifier_df(df, model_out=args.model_out)
        return

    if args.command == "predict_classifier":
        if args.features is None:
            parser.error("--features is required for predict_classifier")
        from ml import predict_win_probability

        feature_values = json.loads(args.features)
        prob = predict_win_probability(args.model, feature_values)
        print(f"Home win probability: {prob:.3f}")
        return

    if args.command == "continuous_train_classifier":
        from ml import continuous_train_classifier

        if not args.start_date:
            parser.error("--start-date is required for continuous_train_classifier")
        continuous_train_classifier(
            args.sport,
            args.start_date,
            interval_hours=args.interval_hours,
            model_out=args.model_out,
        )
        return

    if args.command == "projected_ks_props":
        from datetime import datetime, timedelta

        if args.date:
            target_date = args.date
        else:
            target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

        url = build_odds_url(
            args.sport,
            markets="batter_strikeouts",
            odds_format="american",
        )
        print(
            f"Fetching pitcher K's O/U props for {args.sport} on {target_date}...\n{url}\n"
        )
        odds = fetch_odds(
            args.sport,
            markets="batter_strikeouts",
            odds_format="american",
        )

        results = []
        for game in odds:
            commence_time = game.get("commence_time", "")
            game_date = commence_time[:10] if commence_time else ""
            if game_date != target_date:
                continue
            home = game.get("home_team")
            away = game.get("away_team")
            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker.get("title") or bookmaker.get("key")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "batter_strikeouts":
                        continue
                    pitcher_lines = {}
                    for outcome in market.get("outcomes", []):
                        pitcher = outcome.get("name")
                        line = outcome.get("line")
                        description = outcome.get("description", "").lower()
                        if pitcher is None or line is None:
                            continue
                        key = (pitcher, line)
                        if key not in pitcher_lines:
                            pitcher_lines[key] = {
                                "pitcher": pitcher,
                                "line": line,
                                "price_over": None,
                                "price_under": None,
                            }
                        if description.startswith("over"):
                            pitcher_lines[key]["price_over"] = outcome.get("price")
                        elif description.startswith("under"):
                            pitcher_lines[key]["price_under"] = outcome.get("price")
                    for (pitcher, line), props in pitcher_lines.items():
                        if props["price_over"] is not None and props["price_under"] is not None:
                            results.append(
                                {
                                    "game": f"{home} vs {away}",
                                    "bookmaker": book_name,
                                    "pitcher": pitcher,
                                    "line": line,
                                    "price_over": props["price_over"],
                                    "price_under": props["price_under"],
                                }
                            )

        print(f"Found {len(results)} pitcher K's O/U props for {target_date}")
        for r in results:
            print(
                f"{r['game']} | {r['bookmaker']} | {r['pitcher']} O/U {r['line']} "
                f"(O: {r['price_over']}, U: {r['price_under']})"
            )
        return

    # ---- Projected K's O/U Props w/ ML Model Evaluation ----
    if args.command == "projected_ks_ml_eval":
        from ml import predict_pitcher_ks_over_probability, implied_probability
        from datetime import datetime, timedelta

        # Use today by default, or --date if provided
        if args.date:
            target_date = args.date
        else:
            target_date = (datetime.utcnow()).strftime("%Y-%m-%d")

        url = build_odds_url(
            args.sport,
            markets="batter_strikeouts",
            odds_format="american"
        )
        print(f"Fetching pitcher K's O/U props for {args.sport} on {target_date}...\n{url}\n")

        # --- DEBUG PRINT FOR URL ---
        print(f"[DEBUG] Requesting odds from: {url}")

        try:
            odds = fetch_odds(
                args.sport,
                markets="batter_strikeouts",
                odds_format="american"
            )
        except Exception as e:
            print(f"[DEBUG] Exception while fetching odds: {e}")
            odds = []

        # --- DEBUG PRINT FOR API RESPONSE ---
        print(f"[DEBUG] Response type: {type(odds)}")
        print(
            f"[DEBUG] Raw odds response: {json.dumps(odds, indent=2) if isinstance(odds, (dict, list)) else odds}"
        )

        results = []
        for game in odds:
            commence_time = game.get("commence_time", "")
            game_date = commence_time[:10] if commence_time else ""
            if game_date != target_date:
                continue
            home = game.get("home_team")
            away = game.get("away_team")
            h2h_home, h2h_away = None, None
            # Find moneyline odds for implied win prob
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home:
                                h2h_home = outcome.get("price")
                            elif outcome.get("name") == away:
                                h2h_away = outcome.get("price")
            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker.get("title") or bookmaker.get("key")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "batter_strikeouts":
                        continue
                    pitcher_lines = {}
                    for outcome in market.get("outcomes", []):
                        pitcher = outcome.get("name")
                        line = outcome.get("line")
                        description = outcome.get("description", "").lower()
                        if pitcher is None or line is None:
                            continue
                        key = (pitcher, line)
                        if key not in pitcher_lines:
                            pitcher_lines[key] = {"pitcher": pitcher, "line": line, "price_over": None, "price_under": None}
                        if description.startswith("over"):
                            pitcher_lines[key]["price_over"] = outcome.get("price")
                        elif description.startswith("under"):
                            pitcher_lines[key]["price_under"] = outcome.get("price")
                    for (pitcher, line), props in pitcher_lines.items():
                        # Find implied win prob (matching home/away to pitcher if possible, else None)
                        implied_win_prob = None
                        if pitcher and home and pitcher in home:
                            implied_win_prob = implied_probability(h2h_home) if h2h_home is not None else None
                        elif pitcher and away and pitcher in away:
                            implied_win_prob = implied_probability(h2h_away) if h2h_away is not None else None
                        if props["price_over"] is not None and props["price_under"] is not None and implied_win_prob is not None:
                            features = {
                                "line": line,
                                "price_over": props["price_over"],
                                "price_under": props["price_under"],
                                "implied_win_prob": implied_win_prob
                            }
                            ml_prob = predict_pitcher_ks_over_probability(args.model, features)
                            results.append({
                                "game": f"{home} vs {away}",
                                "bookmaker": book_name,
                                "pitcher": pitcher,
                                "line": line,
                                "price_over": props["price_over"],
                                "price_under": props["price_under"],
                                "ml_prob": ml_prob,
                            })

        print(f"Found {len(results)} pitcher K's O/U props for {target_date} with ML projections")
        for r in results:
            print(
                f"{r['game']} | {r['bookmaker']} | {r['pitcher']} O/U {r['line']} "
                f"(O: {r['price_over']}, U: {r['price_under']}) | ML Prob Over: {r['ml_prob']:.3f}"
            )
        return

    # Odds viewing commands
    markets = "h2h,spreads,totals"
    if args.command == "outrights":
        markets = "outrights"
    elif args.command == "alternate_spreads":
        markets = "alternate_spreads"
    elif args.command == "alternate_totals":
        markets = "alternate_totals"
    elif args.command == "team_totals":
        markets = "team_totals"
    elif args.command == "alternate_team_totals":
        markets = "alternate_team_totals"
    elif args.command == "player_props":
        markets = "player_hits,player_home_runs,player_strikeouts,batter_strikeouts"
    elif args.command == "historical":
        if args.date is None:
            parser.error("--date is required for historical command")

    if args.command == "historical":
        print(f"[DEBUG] Command line args: {args}")
        url = build_historical_odds_url(
            args.sport,
            date=args.date,
            markets=markets,
        )
        print(f"Fetching historical odds for {args.sport} on {args.date}...\n{url}\n")
        odds = fetch_historical_odds(
            args.sport,
            date=args.date,
            markets=markets,
        )
    else:
        url = build_odds_url(
            args.sport,
            markets=markets,
            game_period_markets=args.game_period_markets,
        )
        print(f"Fetching {args.command} odds for {args.sport}...\n{url}\n")
        odds = fetch_odds(
            args.sport,
            markets=markets,
            game_period_markets=args.game_period_markets,
        )

    # If the API response wraps games in a top-level "data" key,
    # extract that list for display. Otherwise assume ``odds`` is
    # already the list of games.
    if isinstance(odds, dict) and "data" in odds:
        games = odds["data"]
    else:
        games = odds

    # Validate the games structure before attempting to format it.
    if (
        not games
        or not isinstance(games, list)
        or not all(isinstance(g, dict) for g in games)
    ):
        # Handle API errors or empty/no-games gracefully
        if isinstance(odds, dict) and odds.get("message"):
            print(f"API message: {odds.get('message')}")
        else:
            print("No odds found or API returned unexpected data.")
        return

    if args.command == "historical":
        print(format_moneyline(games))
    elif args.command == "moneyline":
        print(format_moneyline(games))
    else:
        print(json.dumps(games, indent=2))


if __name__ == "__main__":
    main()
