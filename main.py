import json
import os
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THE_ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set")


def fetch_sports():
    """Return a list of active sports from The Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports?apiKey={API_KEY}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def fetch_odds(sport_key: str, odds_format: str = "american"):
    """Fetch upcoming odds for a given sport."""
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "oddsFormat": odds_format,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def main():
    sports = fetch_sports()
    active = [s for s in sports if s.get("active")]
    print("Active sports:")
    for sport in active:
        print(f"{sport['key']}: {sport['title']}")

    if not active:
        print("No active sports found.")
        return

    key = active[0]["key"]
    print(f"\nFetching odds for {key}...")
    odds = fetch_odds(key)
    print(json.dumps(odds, indent=2)[:1000])


if __name__ == "__main__":
    main()
