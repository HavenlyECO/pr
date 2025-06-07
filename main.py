import os
import json
import urllib.parse
import urllib.request
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
    regions: str = 'us',
    markets: str = 'player_props',
    odds_format: str = 'american',
) -> str:
    """Return the Odds API URL for upcoming markets."""
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': API_KEY,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def fetch_odds(
    sport_key: str,
    *,
    regions: str = 'us',
    markets: str = 'player_props',
    odds_format: str = 'american',
) -> list:
    """Fetch upcoming odds data from the API."""
    url = build_odds_url(sport_key, regions=regions, markets=markets, odds_format=odds_format)
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def tomorrow_iso() -> str:
    """Return tomorrow's date in YYYY-MM-DD (UTC)."""
    return (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')


def evaluate_tomorrows_player_strikeouts(
    sport_key: str,
    model_path: str,
    *,
    regions: str = 'us',
) -> list:
    """Return strikeout prop evaluations for games starting tomorrow."""
    from ml import predict_pitcher_ks_over_probability

    odds = fetch_odds(sport_key, regions=regions, markets='player_props')
    target_date = tomorrow_iso()
    results = []

    for game in odds:
        commence = game.get('commence_time', '')
        if not commence.startswith(target_date):
            continue
        home = game.get('home_team')
        away = game.get('away_team')
        for book in game.get('bookmakers', []):
            book_name = book.get('title') or book.get('key')
            for market in book.get('markets', []):
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
                        line_map[key] = {'player': player, 'line': line, 'price_over': None, 'price_under': None}
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
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Display projected player strikeout props for tomorrow.'
    )
    parser.add_argument('--sport', default='baseball_mlb', help='Sport key')
    parser.add_argument('--regions', default='us', help='Comma separated regions (default: us)')
    parser.add_argument('--model', default='pitcher_ks_classifier.pkl', help='Path to trained ML model')
    args = parser.parse_args()

    projections = evaluate_tomorrows_player_strikeouts(
        args.sport,
        args.model,
        regions=args.regions,
    )
    print(json.dumps(projections, indent=2))


if __name__ == '__main__':
    main()
