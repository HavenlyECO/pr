import os
import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
import argparse

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required. Install it with 'pip install python-dotenv'")

ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / '.env'
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

API_KEY = os.getenv('THE_ODDS_API_KEY')
if not API_KEY:
    raise RuntimeError('THE_ODDS_API_KEY environment variable is not set')


def tomorrow_iso() -> str:
    return (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')


def build_events_url(sport_key: str, regions: str = "us") -> str:
    return (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
        f"?apiKey={API_KEY}&regions={regions}"
    )


def fetch_events(sport_key: str, regions: str = "us") -> list:
    url = build_events_url(sport_key, regions=regions)
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def build_event_odds_url(
    sport_key: str,
    event_id: str,
    markets: str = "batter_strikeouts",
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
    player_props: bool = True,
) -> str:
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if player_props:
        params["playerProps"] = "true"
    return f"{base}?{urllib.parse.urlencode(params)}"


def fetch_event_odds(
    sport_key: str,
    event_id: str,
    markets: str = "batter_strikeouts",
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
    player_props: bool = True,
) -> list:
    url = build_event_odds_url(
        sport_key,
        event_id,
        markets=markets,
        regions=regions,
        odds_format=odds_format,
        date_format=date_format,
        player_props=player_props,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError:
        return []


def evaluate_batter_strikeouts_all_tomorrow(
    sport_key: str,
    model_path: str,
    regions: str = "us"
) -> list:
    from ml import predict_pitcher_ks_over_probability

    events = fetch_events(sport_key, regions=regions)
    target_date = tomorrow_iso()
    results = []

    for event in events:
        commence = event.get('commence_time', '')
        if not commence.startswith(target_date):
            continue
        event_id = event.get('id')
        home = event.get('home_team')
        away = event.get('away_team')
        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets="batter_strikeouts",
            regions=regions
        )
        for game in game_odds:
            for book in game.get('bookmakers', []):
                book_name = book.get('title') or book.get('key')
                for market in book.get('markets', []):
                    if market.get('key') != 'batter_strikeouts':
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
                            'event_id': event_id,
                            'projected_over_probability': prob,
                        })
    return results


def print_projections_table(projections: list) -> None:
    if not projections:
        print("No projection data available.")
        return

    headers = [
        "GAME",
        "BOOK",
        "PLAYER",
        "LINE",
        "OVER",
        "UNDER",
        "P(OVER)",
    ]

    def col_width(key, min_width):
        return max(min_width, max(len(str(row.get(key, ""))) for row in projections))

    widths = {
        "GAME": col_width("game", 10),
        "BOOK": col_width("bookmaker", 6),
        "PLAYER": col_width("player", 8),
        "LINE": col_width("line", 4),
        "OVER": col_width("price_over", 4),
        "UNDER": col_width("price_under", 5),
        "P(OVER)": 7,
    }

    header_line = " ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in projections:
        prob = row.get("projected_over_probability")
        prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"
        values = [
            row.get("game", ""),
            row.get("bookmaker", ""),
            row.get("player", ""),
            row.get("line", ""),
            row.get("price_over", ""),
            row.get("price_under", ""),
            prob_str,
        ]
        print(" ".join(str(v).ljust(widths[h]) for v, h in zip(values, headers)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Display projected batter strikeout props for tomorrow (autofetch event IDs).'
    )
    parser.add_argument('--sport', default='baseball_mlb', help='Sport key')
    parser.add_argument('--regions', default='us', help='Comma separated regions (default: us)')
    parser.add_argument('--model', default='pitcher_ks_classifier.pkl', help='Path to trained ML model')
    args = parser.parse_args()

    projections = evaluate_batter_strikeouts_all_tomorrow(
        args.sport,
        args.model,
        regions=args.regions,
    )
    print_projections_table(projections)


if __name__ == '__main__':
    main()
