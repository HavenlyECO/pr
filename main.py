import os
import json
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
import argparse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - guidance for missing dependency
    raise ImportError("python-dotenv is required. Install it with 'pip install python-dotenv'")

# Load environment variables from project root
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / '.env'
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

API_KEY = os.getenv('THE_ODDS_API_KEY')
if not API_KEY:
    raise RuntimeError('THE_ODDS_API_KEY environment variable is not set')


def build_odds_url(
    sport_key: str,
    *,
    event_id: str | None = None,
    regions: str = 'us',
    markets: str = 'batter_strikeouts',
    odds_format: str = 'american',
    date_format: str = 'iso',
    player_props: bool = False,
) -> str:
    """Return the Odds API URL for upcoming markets or a specific event."""
    if event_id:
        base = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
        )
    else:
        base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': API_KEY,
        'regions': regions,
        'oddsFormat': odds_format,
    }
    if markets:
        params['markets'] = markets
    if event_id:
        params['dateFormat'] = date_format
    if player_props:
        params['playerProps'] = 'true'
    return f"{base}?{urllib.parse.urlencode(params)}"


def build_events_url(sport_key: str) -> str:
    """Return the Odds API URL for upcoming events for a sport."""
    return f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?apiKey={API_KEY}"


def fetch_events(sport_key: str) -> list:
    """Fetch upcoming events for the given sport."""
    url = build_events_url(sport_key)
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode()
            except Exception:
                body = str(e.read())
        error_msg = f"HTTPError fetching events: {e.code} {e.reason}\n{body}\nURL: {url}"
        raise RuntimeError(error_msg) from e


def fetch_odds(
    sport_key: str,
    *,
    event_id: str | None = None,
    regions: str = "us",
    markets: str = "batter_strikeouts",
    odds_format: str = "american",
    date_format: str = "iso",
    player_props: bool = False,
) -> list:
    """Fetch odds data from the API."""
    url = build_odds_url(
        sport_key,
        event_id=event_id,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        date_format=date_format,
        player_props=player_props,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode()
            except Exception:
                body = str(e.read())
        error_msg = (
            f"HTTPError fetching odds: {e.code} {e.reason}\n{body}\nURL: {url}"
        )
        raise RuntimeError(error_msg) from e


def tomorrow_iso() -> str:
    """Return tomorrow's date in YYYY-MM-DD (UTC)."""
    return (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')


def evaluate_tomorrows_strikeout_props(
    sport_key: str,
    model_path: str,
    *,
    event_id: str | None = None,
    regions: str = 'us',
    markets: str = 'batter_strikeouts',
    player_props: bool = False,
) -> list:
    """Return strikeout prop evaluations for games starting tomorrow.

    This fetches odds for tomorrow's games, filters for strikeout props, and
    evaluates the probability of the over using the ML model. If ``player_props``
    is ``True`` the request includes player prop markets.
    """
    from ml import predict_pitcher_ks_over_probability

    odds = fetch_odds(
        sport_key,
        event_id=event_id,
        regions=regions,
        markets=markets,
        player_props=player_props,
    )
    print("RAW ODDS DATA:", json.dumps(odds, indent=2))
    target_date = tomorrow_iso()
    results = []
    found_any_game = False

    for game in odds:
        commence = game.get('commence_time', '')
        if event_id is None and not commence.startswith(target_date):
            continue
        found_any_game = True
        print(
            "Found game:",
            game.get('home_team'),
            "vs",
            game.get('away_team'),
            "at",
            commence,
        )
        home = game.get('home_team')
        away = game.get('away_team')
        for book in game.get('bookmakers', []):
            book_name = book.get('title') or book.get('key')
            for market in book.get('markets', []):
                print(
                    "MARKET KEY:",
                    market.get('key'),
                    "| DESC:",
                    market.get('description', ''),
                )
                is_strikeout_market = (
                    'strikeout' in market.get('key', '').lower()
                    or 'strikeout' in market.get('description', '').lower()
                )
                if not is_strikeout_market:
                    continue
                line_map = {}
                for outcome in market.get('outcomes', []):
                    player = outcome.get('name')
                    line = outcome.get('line')
                    desc = outcome.get('description', '').lower()
                    if player is None or line is None:
                        continue
                    key = (player, line)
                    if key not in line_map:
                        line_map[key] = {
                            'player': player,
                            'line': line,
                            'price_over': None,
                            'price_under': None,
                        }
                    if desc.startswith('over'):
                        line_map[key]['price_over'] = outcome.get('price')
                    elif desc.startswith('under'):
                        line_map[key]['price_under'] = outcome.get('price')
                for props in line_map.values():
                    if props['price_over'] is None or props['price_under'] is None:
                        continue
                    features = {
                        'line': props['line'],
                        'price_over': props['price_over'],
                        'price_under': props['price_under'],
                    }
                    prob = predict_pitcher_ks_over_probability(model_path, features)
                    results.append({
                        'game': f"{home} vs {away}",
                        'bookmaker': book_name,
                        'player': props['player'],
                        'line': props['line'],
                        'price_over': props['price_over'],
                        'price_under': props['price_under'],
                        'projected_over_probability': prob,
                    })
    if not found_any_game:
        print("No games found for tomorrow.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Display projected pitcher strikeout props for tomorrow.'
    )
    parser.add_argument('--sport', default='baseball_mlb', help='Sport key')
    parser.add_argument('--regions', default='us', help='Comma separated regions (default: us)')
    parser.add_argument('--model', default='pitcher_ks_classifier.pkl', help='Path to trained ML model')
    parser.add_argument(
        '--markets',
        default='batter_strikeouts',
        help='Comma separated markets (default: batter_strikeouts)'
    )
    parser.add_argument(
        '--event-id',
        help='Specific event ID to fetch odds for (optional)'
    )
    parser.add_argument(
        '--player-props',
        action='store_true',
        help='Include player props in the event odds request'
    )
    parser.add_argument(
        '--list-events',
        action='store_true',
        help='List upcoming events for the given sport and exit'
    )
    args = parser.parse_args()

    if args.list_events:
        events = fetch_events(args.sport)
        print(json.dumps(events, indent=2))
        return

    projections = evaluate_tomorrows_strikeout_props(
        args.sport,
        args.model,
        event_id=args.event_id,
        regions=args.regions,
        markets=args.markets,
        player_props=args.player_props,
    )
    print(json.dumps(projections, indent=2))


if __name__ == '__main__':
    main()
