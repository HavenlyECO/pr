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
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        message = e.read().decode() if hasattr(e, "read") else str(e)
        try:
            msg_json = json.loads(message)
            if msg_json.get("error_code") == "INVALID_HISTORICAL_TIMESTAMP":
                print(
                    f"\n[!] No historical data available for {sport_key} on {date}: {msg_json.get('message')}"
                )
                print(
                    "    - This usually means the date is out of range, not yet available, or off-season."
                )
                print("    - Try a different, more recent or in-season date.\n")
                return []
        except Exception:
            pass
        print(f"HTTPError fetching odds: {message}")
        return []
    except Exception as e:
        print(f"Error fetching historical odds: {e}")
        return []


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
        markets = "player_hits,player_home_runs,player_strikeouts"
    elif args.command == "historical":
        if args.date is None:
            parser.error("--date is required for historical command")

    if args.command == "historical":
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

    if not odds:
        print("No odds found for your query.")
        return

    if args.command in {"moneyline", "historical"}:
        print(format_moneyline(odds))
    else:
        print(json.dumps(odds, indent=2))


if __name__ == "__main__":
    main()
