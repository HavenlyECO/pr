import argparse
import json
import os
import urllib.parse
import urllib.request
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        """Fallback no-op if python-dotenv isn't installed."""
        pass

load_dotenv()

API_KEY = os.getenv("THE_ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set")


def build_odds_url(
    sport_key: str,
    *,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    game_period_markets: str | None = None,
) -> str:
    """Return the fully qualified odds API URL."""
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
    """Fetch upcoming odds for a given sport.

    Parameters
    ----------
    sport_key: str
        The key of the sport (e.g. ``"soccer_epl"``).
    regions: str, optional
        Comma separated region codes. Defaults to ``"us"``.
    markets: str, optional
        Market types to return, e.g. ``"h2h,spreads"``. Defaults to ``"h2h"``.
    odds_format: str, optional
        Format of the odds to return. Defaults to ``"american"``.
    """

    url = build_odds_url(
        sport_key,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        game_period_markets=game_period_markets,
    )
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def build_historical_odds_url(
    sport_key: str,
    *,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    include_scores: bool = False,
) -> str:
    """Return the fully qualified historical odds API URL."""
    base_url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds-history"
    )
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
    except urllib.error.HTTPError as e:  # pragma: no cover - network error handling
        message = e.read().decode() if hasattr(e, "read") else str(e)
        raise RuntimeError(f"Failed to fetch historical odds: {message}") from e


def _format_header(idx: int, game: dict) -> str:
    home = game.get("home_team", "N/A")
    away = game.get("away_team", "N/A")
    time = game.get("commence_time", "")
    return f"{idx}. {home} vs {away} ({time})"


def format_moneyline(games):
    """Return a formatted string of moneyline odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Head-to-Head (Moneyline):")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('price', '')}"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def format_spreads(games):
    """Return a formatted string of point spread odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Spreads (Handicap):")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "spreads":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('point', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def format_totals(games):
    """Return a formatted string of totals (over/under) odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Totals (Over/Under):")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('point', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def format_alternate_spreads(games):
    """Return a formatted string of alternate spread odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Alternate Spreads:")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "alternate_spreads":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('point', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def format_alternate_totals(games):
    """Return a formatted string of alternate totals odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Alternate Totals:")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "alternate_totals":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('point', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def format_team_totals(games):
    """Return a formatted string of team totals odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Team Totals:")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "team_totals":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('point', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def format_alternate_team_totals(games):
    """Return a formatted string of alternate team totals odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Alternate Team Totals:")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "alternate_team_totals":
                    continue
                outcomes = [
                    f"{o.get('name', '')} {o.get('point', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


PLAYER_PROP_MARKETS = {
    "player_hits": "Player Hits",
    "player_home_runs": "Player Home Runs",
    "player_strikeouts": "Player Strikeouts",
}


def format_player_props(games):
    """Return a formatted string of player prop odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        lines.append(_format_header(idx, game))
        lines.append("   Player Props:")

        for bookmaker in game.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                if key not in PLAYER_PROP_MARKETS:
                    continue
                outcomes = []
                for o in market.get("outcomes", []):
                    point = o.get("point")
                    if point is not None:
                        outcomes.append(
                            f"{o.get('name', '')} {point} ({o.get('price', '')})"
                        )
                    else:
                        outcomes.append(f"{o.get('name', '')} ({o.get('price', '')})")
                if outcomes:
                    lines.append(
                        f"      {bm_title} {PLAYER_PROP_MARKETS[key]}: "
                        + " | ".join(outcomes)
                    )
        lines.append("")

    return "\n".join(lines)


def _format_outright_header(idx: int, event: dict) -> str:
    title = event.get("title", "Outright")
    time = event.get("commence_time", "")
    return f"{idx}. {title} ({time})"


def format_outrights(events):
    """Return a formatted string of outright (futures) odds for display."""
    lines = []
    for idx, event in enumerate(events, 1):
        lines.append(_format_outright_header(idx, event))
        lines.append("   Futures:")

        for bookmaker in event.get("bookmakers", []):
            bm_title = bookmaker.get("title", bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                outcomes = [
                    f"{o.get('name', '')} ({o.get('price', '')})"
                    for o in market.get("outcomes", [])
                ]
                if outcomes:
                    lines.append(f"      {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def main():
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
    parser.add_argument(
        "--sport",
        default="baseball_mlb",
        help="Sport key, e.g. baseball_mlb",
    )
    parser.add_argument(
        "--game-period-markets",
        default=None,
        help="Comma separated game period markets to include, e.g. first_half_totals",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="CSV file for training the moneyline classifier",
    )
    parser.add_argument(
        "--model-out",
        default="moneyline_classifier.pkl",
        help="Where to save the trained classifier",
    )
    parser.add_argument(
        "--model",
        default="moneyline_classifier.pkl",
        help="Path to a trained classifier for prediction",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="JSON string of feature values for prediction",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date for historical odds in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date for training data in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for training data in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=24,
        help="Interval between training runs in hours for continuous training",
    )
    args = parser.parse_args()

    if args.command == "train_classifier":
        from ml import (
            train_classifier,
            train_classifier_df,
            build_dataset_from_api,
        )

        if args.dataset:
            train_classifier(args.dataset, model_out=args.model_out)
        else:
            if not args.start_date or not args.end_date:
                parser.error(
                    "--dataset or --start-date/--end-date required for train_classifier"
                )
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
        print(
            f"Fetching historical odds for {args.sport} on {args.date}...\n{url}\n"
        )
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
        print("No odds found.")
        return

    if args.command == "moneyline":
        print(format_moneyline(odds))
    elif args.command == "spreads":
        print(format_spreads(odds))
    elif args.command == "totals":
        print(format_totals(odds))
    elif args.command == "alternate_spreads":
        print(format_alternate_spreads(odds))
    elif args.command == "alternate_totals":
        print(format_alternate_totals(odds))
    elif args.command == "team_totals":
        print(format_team_totals(odds))
    elif args.command == "alternate_team_totals":
        print(format_alternate_team_totals(odds))
    elif args.command == "player_props":
        print(format_player_props(odds))
    elif args.command == "historical":
        print(format_moneyline(odds))
    else:
        print(format_outrights(odds))


if __name__ == "__main__":
    main()
