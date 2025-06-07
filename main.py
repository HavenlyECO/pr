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
    except urllib.error.HTTPError as e:
        # Optionally handle/log errors here
        return []
    except Exception as e:
        # Defensive: Return empty on any other error
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

    print(f"DEBUG: {len(events)} events returned by API")
    for event in events:
        commence = event.get('commence_time', '')
        event_id = event.get('id')
        home = event.get('home_team')
        away = event.get('away_team')
        print(f"\nEVENT: {event_id} | {away} at {home} | {commence}")

        if not commence.startswith(target_date):
            print("  Skipped: does not match target date")
            continue

        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets="batter_strikeouts",
            regions=regions
        )
        print(f"  Raw odds for event {event_id}:")
        print(json.dumps(game_odds, indent=2))

        # Handle if the API returns a single dict (with 'bookmakers') or a list
        if isinstance(game_odds, dict) and 'bookmakers' in game_odds:
            game_odds = [game_odds]
        elif not isinstance(game_odds, list):
            print(f"  Skipped: unexpected odds format: {type(game_odds)} {game_odds}")
            continue

        for game in game_odds:
            if not isinstance(game, dict):
                print(f"  Skipped: game is not a dict: {game}")
                continue
            if not game.get('bookmakers'):
                print(f"  Skipped: no bookmakers in game {game.get('id')} for this event")
                continue  # no props posted for this event
            print(f"  Bookmakers found: {[b.get('title') or b.get('key') for b in game.get('bookmakers',[])]}")

            for book in game.get('bookmakers', []):
                book_name = book.get('title') or book.get('key')
                print(f"    Bookmaker: {book_name}")
                if not book.get('markets'):
                    print("      Skipped: no markets in this bookmaker")
                    continue
                for market in book.get('markets', []):
                    print(f"      Market key: {market.get('key')}, desc: {market.get('description')}")
                    if market.get('key') != 'batter_strikeouts':
                        print("        Skipped: not a batter_strikeouts market")
                        continue
                    if not market.get('outcomes'):
                        print("        Skipped: no outcomes in market")
                        continue
                    line_map = {}
                    for outcome in market.get('outcomes', []):
                        player = outcome.get('name')
                        line = outcome.get('line')
                        desc = outcome.get('description', '').lower()
                        if player is None or line is None:
                            print(f"        Skipped outcome: missing player or line: {outcome}")
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
                            print(f"        Skipped: missing price_over or price_under for {props}")
                            continue
                        features = {
                            'line': props['line'],
                            'price_over': props['price_over'],
                            'price_under': props['price_under'],
                        }
                        prob = predict_pitcher_ks_over_probability(model_path, features)
                        print(
                            f"        EVAL: {props['player']} line={props['line']} over={props['price_over']} under={props['price_under']} prob={prob}"
                        )
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
    print(f"DEBUG: Total evaluated props: {len(results)}")
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


def print_event_odds(
    sport_key: str,
    event_id: str,
    markets: str,
    regions: str,
    odds_format: str,
    date_format: str,
    player_props: bool,
) -> None:
    """Fetch odds for a single event and print the raw JSON."""

    game_odds = fetch_event_odds(
        sport_key,
        event_id,
        markets=markets,
        regions=regions,
        odds_format=odds_format,
        date_format=date_format,
        player_props=player_props,
    )
    print(json.dumps(game_odds, indent=2))


def list_market_keys(
    sport_key: str,
    markets: str,
    regions: str,
    odds_format: str,
    date_format: str,
    player_props: bool,
    game_period_markets: str | None = None,
) -> None:
    """List all market keys available for upcoming games."""

    events = fetch_events(sport_key, regions=regions)
    if not events:
        print("No upcoming events found.")
        return

    req_markets = markets
    if game_period_markets:
        req_markets = f"{markets},{game_period_markets}" if markets else game_period_markets

    for event in events:
        event_id = event.get("id")
        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets=req_markets,
            regions=regions,
            odds_format=odds_format,
            date_format=date_format,
            player_props=player_props,
        )
        print(json.dumps(game_odds, indent=2))
        if isinstance(game_odds, list):
            for game in game_odds:
                for book in game.get("bookmakers", []):
                    for market in book.get("markets", []):
                        print(
                            f"Market key: {market.get('key')}, desc: {market.get('description')}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Display projected batter strikeout props for tomorrow (autofetch event IDs).'
    )
    parser.add_argument('--sport', default='baseball_mlb', help='Sport key')
    parser.add_argument('--regions', default='us', help='Comma separated regions (default: us)')
    parser.add_argument('--model', default='pitcher_ks_classifier.pkl', help='Path to trained ML model')
    parser.add_argument('--markets', default='batter_strikeouts', help='Comma separated market keys')
    parser.add_argument('--odds-format', default='american', help='Odds format')
    parser.add_argument('--date-format', default='iso', help='Date format')
    parser.add_argument('--event-id', help='Event ID for event odds request')
    parser.add_argument('--event-odds', action='store_true', help='Print raw odds for the given event ID and exit')
    parser.add_argument('--list-market-keys', action='store_true', help='List market keys for upcoming events and exit')
    parser.add_argument('--game-period-markets', help='Comma separated game period market keys to include')
    parser.add_argument('--no-player-props', action='store_true', help='Exclude player prop markets')
    parser.add_argument(
        '--list-events',
        action='store_true',
        help='List upcoming events for the given sport and exit'
    )
    args = parser.parse_args()

    if args.event_odds:
        if not args.event_id:
            print('--event-id is required with --event-odds')
            return
        req_markets = args.markets
        if args.game_period_markets:
            req_markets = f"{args.markets},{args.game_period_markets}" if args.markets else args.game_period_markets
        print_event_odds(
            args.sport,
            args.event_id,
            req_markets,
            regions=args.regions,
            odds_format=args.odds_format,
            date_format=args.date_format,
            player_props=not args.no_player_props,
        )
        return

    if args.list_market_keys:
        list_market_keys(
            args.sport,
            args.markets,
            regions=args.regions,
            odds_format=args.odds_format,
            date_format=args.date_format,
            player_props=not args.no_player_props,
            game_period_markets=args.game_period_markets,
        )
        return

    if args.list_events:
        events = fetch_events(args.sport, regions=args.regions)
        if not events:
            print('No upcoming events found.')
            return
        for event in events:
            commence = event.get('commence_time', 'N/A')
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            event_id = event.get('id', '')
            print(f"{commence} - {away} at {home} ({event_id})")
        return

    projections = evaluate_batter_strikeouts_all_tomorrow(
        args.sport,
        args.model,
        regions=args.regions,
    )
    print_projections_table(projections)


if __name__ == '__main__':
    main()
