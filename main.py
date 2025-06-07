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
    markets: str = '',
    odds_format: str = 'american',
    date_format: str = 'iso',
    player_props: bool = True,
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


def build_events_url(sport_key: str, *, regions: str = "us") -> str:
    """Return the Odds API URL for upcoming events for a sport."""
    return (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
        f"?apiKey={API_KEY}&regions={regions}"
    )


def fetch_events(sport_key: str, *, regions: str = "us") -> list:
    """Fetch upcoming events for the given sport."""
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


def fetch_odds(
    sport_key: str,
    *,
    event_id: str | None = None,
    regions: str = "us",
    markets: str = "",
    odds_format: str = "american",
    date_format: str = "iso",
    player_props: bool = True,
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
    markets: str = '',
    player_props: bool = True,
    collect_h2h: bool = False,
) -> list | tuple[list, dict]:
    """Return strikeout prop evaluations for games starting tomorrow.

    This fetches odds for tomorrow's games, filters for strikeout props, and
    evaluates the probability of the over using the ML model. If ``player_props``
    is ``True`` the request includes player prop markets. Set ``collect_h2h`` to
    ``True`` to also gather head-to-head prices for each game/bookmaker.
    """
    from ml import predict_pitcher_ks_over_probability

    odds = fetch_odds(
        sport_key,
        event_id=event_id,
        regions=regions,
        markets=markets,
        player_props=player_props,
    )
    # The raw API response can be extremely verbose which makes it hard to
    # view all player props on a single screen.  Comment the next line in when
    # debugging to inspect the full JSON payload.
    # print("RAW ODDS DATA:", json.dumps(odds, indent=2))
    target_date = tomorrow_iso()
    results = []
    if collect_h2h:
        h2h_data: dict[tuple[str, str], dict] = {}
    else:
        h2h_data = {}
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
            if collect_h2h:
                for market in book.get('markets', []):
                    if market.get('key') == 'h2h':
                        price_home = None
                        price_away = None
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home:
                                price_home = outcome.get('price')
                            elif outcome.get('name') == away:
                                price_away = outcome.get('price')
                        if price_home is not None and price_away is not None:
                            h2h_data[(game.get('id'), book_name)] = {
                                'home_team': home,
                                'away_team': away,
                                'price_home': price_home,
                                'price_away': price_away,
                            }
                        break
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
                        'event_id': game.get('id'),
                        'projected_over_probability': prob,
                    })
    if not found_any_game:
        print("No games found for tomorrow.")
    if collect_h2h:
        return results, h2h_data
    return results


def collect_market_keys(
    sport_key: str,
    *,
    event_id: str | None = None,
    regions: str = "us",
    markets: str = "",
    player_props: bool = True,
) -> list[tuple[str | None, str | None]]:
    """Return unique market (key, description) pairs for upcoming games."""
    odds = fetch_odds(
        sport_key,
        event_id=event_id,
        regions=regions,
        markets=markets,
        player_props=player_props,
    )
    pairs: set[tuple[str | None, str | None]] = set()
    for game in odds:
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                pairs.add(
                    (
                        market.get("key"),
                        market.get("description"),
                    )
                )
    return sorted(pairs)


def print_projections_table(projections: list) -> None:
    """Display projection results in a compact table."""
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

    # Determine column widths dynamically so most results fit on one line
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


def print_h2h_with_projections(projections: list, h2h_data: dict) -> None:
    """Display H2H lines with strikeout projections grouped underneath."""
    if not projections:
        print("No projection data available.")
        return

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in projections:
        key = (row.get('game', ''), row.get('bookmaker', ''))
        grouped.setdefault(key, []).append(row)

    for (game, book), items in grouped.items():
        event_id = items[0].get('event_id')
        print(f"{game} - {book}")
        h2h = h2h_data.get((event_id, book))
        if h2h:
            home = h2h['home_team']
            away = h2h['away_team']
            print(
                f"  H2H: {home} {h2h['price_home']} vs {away} {h2h['price_away']}"
            )
        for row in items:
            prob = row.get('projected_over_probability')
            prob_str = f"{prob*100:.1f}%" if prob is not None else 'N/A'
            print(
                f"  {row['player']} {row['line']} O {row['price_over']} U {row['price_under']} P(OVER) {prob_str}"
            )
        print()

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Display projected pitcher strikeout props for tomorrow.'
    )
    parser.add_argument('--sport', default='baseball_mlb', help='Sport key')
    parser.add_argument('--regions', default='us', help='Comma separated regions (default: us)')
    parser.add_argument('--model', default='pitcher_ks_classifier.pkl', help='Path to trained ML model')
    parser.add_argument(
        '--markets',
        default='',
        help='Comma separated markets (empty for all)'
    )
    parser.add_argument(
        '--event-id',
        help='Specific event ID to fetch odds for (optional)'
    )
    parser.add_argument(
        '--player-props',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Include player props in the event odds request'
    )
    parser.add_argument(
        '--list-events',
        action='store_true',
        help='List upcoming events for the given sport and exit'
    )
    parser.add_argument(
        '--list-market-keys',
        action='store_true',
        help='List unique market keys/descriptions for upcoming games and exit'
    )
    parser.add_argument(
        '--blend-h2h',
        action='store_true',
        help='Display H2H odds with projected strikeout props underneath'
    )
    args = parser.parse_args()

    if args.list_events:
        events = fetch_events(args.sport, regions=args.regions)
        print(json.dumps(events, indent=2))
        return

    if args.list_market_keys:
        pairs = collect_market_keys(
            args.sport,
            event_id=args.event_id,
            regions=args.regions,
            markets=args.markets,
            player_props=args.player_props,
        )
        for key, desc in pairs:
            if desc:
                print(f"{key} - {desc}")
            else:
                print(key)
        return

    projections_data = evaluate_tomorrows_strikeout_props(
        args.sport,
        args.model,
        event_id=args.event_id,
        regions=args.regions,
        markets=args.markets,
        player_props=args.player_props,
        collect_h2h=args.blend_h2h,
    )
    if args.blend_h2h:
        projections, h2h_lines = projections_data
        print_h2h_with_projections(projections, h2h_lines)
    else:
        print_projections_table(projections_data)


if __name__ == '__main__':
    main()
