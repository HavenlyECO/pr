import argparse
import json
import os
import urllib.parse
import urllib.request
from dotenv import load_dotenv

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
) -> str:
    """Return the fully qualified odds API URL."""
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_odds(
    sport_key: str,
    *,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
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
    )
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


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


def main():
    parser = argparse.ArgumentParser(description="Fetch and display odds")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["moneyline", "spreads", "totals"],
        default="moneyline",
        help="Type of odds to display",
    )
    parser.add_argument(
        "--sport",
        default="baseball_mlb",
        help="Sport key, e.g. baseball_mlb",
    )
    args = parser.parse_args()

    markets = "h2h,spreads,totals"
    url = build_odds_url(args.sport, markets=markets)
    print(f"Fetching {args.command} odds for {args.sport}...\n{url}\n")
    odds = fetch_odds(args.sport, markets=markets)

    if not odds:
        print("No odds found.")
        return

    if args.command == "moneyline":
        print(format_moneyline(odds))
    elif args.command == "spreads":
        print(format_spreads(odds))
    else:
        print(format_totals(odds))


if __name__ == "__main__":
    main()
