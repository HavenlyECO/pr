import json
import os
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THE_ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set")


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

    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def format_odds(games):
    """Return a formatted string of odds for display."""
    lines = []
    for idx, game in enumerate(games, 1):
        home = game.get("home_team", "N/A")
        away = game.get("away_team", "N/A")
        time = game.get("commence_time", "")
        lines.append(f"{idx}. {home} vs {away} ({time})")

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
                    lines.append(f"   {bm_title}: " + " | ".join(outcomes))
                break
        lines.append("")

    return "\n".join(lines)


def main():
    key = "baseball_mlb"
    print(f"Fetching odds for {key}...\n")
    odds = fetch_odds(key, markets="h2h,spreads")

    if not odds:
        print("No odds found.")
        return

    print(format_odds(odds))


if __name__ == "__main__":
    main()
