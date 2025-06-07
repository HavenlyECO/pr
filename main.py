import os
import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# For improved table and color output
try:
    from tabulate import tabulate
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print(
        "Please install tabulate and colorama for improved output: pip install tabulate colorama"
    )
    tabulate = None
    Fore = Style = None

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv is required. Install it with 'pip install python-dotenv'"
    )

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
        print(
            Fore.RED + f"HTTPError fetching event odds: {e.code} {e.reason} for URL: {url}"
            if Fore
            else f"HTTPError fetching event odds: {e.code} {e.reason} for URL: {url}"
        )
        return []
    except Exception as e:
        print(
            Fore.RED + f"Error fetching event odds: {e}"
            if Fore
            else f"Error fetching event odds: {e}"
        )
        return []


def evaluate_batter_strikeouts_all_tomorrow(
    sport_key: str,
    model_path: str,
    regions: str = "us"
) -> list:
    from ml import predict_pitcher_ks_over_probability

    events = fetch_events(sport_key, regions=regions)
    results = []

    print(
        Fore.CYAN + f"DEBUG: {len(events)} events returned by API"
        if Fore
        else f"DEBUG: {len(events)} events returned by API"
    )
    for event in events:
        commence = event.get('commence_time', '')
        event_id = event.get('id')
        home = event.get('home_team')
        away = event.get('away_team')
        print(
            f"\n{Style.BRIGHT if Style else ''}EVENT: {event_id} | {away} at {home} | {commence}"
        )

        try:
            commence_dt = datetime.strptime(commence, "%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            print(
                Fore.YELLOW + f"  Skipped: invalid commence_time format {commence} ({e})"
                if Fore
                else f"  Skipped: invalid commence_time format {commence} ({e})"
            )
            continue

        today = datetime.utcnow()
        start_dt = datetime(today.year, today.month, today.day, 16, 0, 0)
        end_dt = start_dt + timedelta(hours=14)  # 16:00 today to 06:00 tomorrow

        if not (start_dt <= commence_dt < end_dt):
            print(
                Fore.YELLOW
                + f"  Skipped: commence_time {commence_dt} not in extended window {start_dt} to {end_dt}"
                if Fore
                else f"  Skipped: commence_time {commence_dt} not in extended window {start_dt} to {end_dt}"
            )
            continue

        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets="batter_strikeouts",
            regions=regions,
        )

        # Debug: show all bookmaker keys available for this event (even if no markets are present)
        all_bookmaker_keys = []
        if isinstance(game_odds, dict) and 'bookmakers' in game_odds:
            all_bookmaker_keys = [b.get('key') for b in game_odds['bookmakers']]
            game_odds = [game_odds]
        elif isinstance(game_odds, list):
            for game in game_odds:
                all_bookmaker_keys.extend([b.get('key') for b in game.get('bookmakers', [])])
        print(
            Fore.MAGENTA + f"  Bookmaker keys in event odds: {sorted(set(all_bookmaker_keys))}"
            if Fore
            else f"  Bookmaker keys in event odds: {sorted(set(all_bookmaker_keys))}"
        )

        if not isinstance(game_odds, list):
            print(
                Fore.RED
                + f"  Skipped: unexpected odds format: {type(game_odds)} {game_odds}"
                if Fore
                else f"  Skipped: unexpected odds format: {type(game_odds)} {game_odds}"
            )
            continue

        for game in game_odds:
            if not isinstance(game, dict):
                print(
                    Fore.RED + f"  Skipped: game is not a dict: {game}"
                    if Fore
                    else f"  Skipped: game is not a dict: {game}"
                )
                continue
            if not game.get('bookmakers'):
                print(
                    Fore.YELLOW
                    + f"  Skipped: no bookmakers in game {game.get('id')} for this event"
                    if Fore
                    else f"  Skipped: no bookmakers in game {game.get('id')} for this event"
                )
                continue  # no props posted for this event
            print(
                Fore.GREEN
                + f"  Bookmakers found: {[b.get('title') or b.get('key') for b in game.get('bookmakers',[])]}"
                if Fore
                else f"  Bookmakers found: {[b.get('title') or b.get('key') for b in game.get('bookmakers',[])]}"
            )

            for book in game.get('bookmakers', []):
                book_name = book.get('title') or book.get('key')
                print(f"    Bookmaker: {book_name}")
                if not book.get('markets'):
                    print(
                        Fore.YELLOW + "      Skipped: no markets in this bookmaker"
                        if Fore
                        else "      Skipped: no markets in this bookmaker"
                    )
                    continue
                for market in book.get('markets', []):
                    print(
                        f"      Market key: {market.get('key')}, desc: {market.get('description')}"
                    )
                    if market.get('key') != 'batter_strikeouts':
                        print(
                            Fore.YELLOW + "        Skipped: not a batter_strikeouts market"
                            if Fore
                            else "        Skipped: not a batter_strikeouts market"
                        )
                        continue
                    if not market.get('outcomes'):
                        print(
                            Fore.YELLOW + "        Skipped: no outcomes in market"
                            if Fore
                            else "        Skipped: no outcomes in market"
                        )
                        continue
                    line_map = {}
                    for outcome in market.get('outcomes', []):
                        player = outcome.get('name')
                        line = outcome.get('line')
                        desc = outcome.get('description', '').lower()
                        if player is None or line is None:
                            print(
                                Fore.RED
                                + f"        Skipped outcome: missing player or line: {outcome}"
                                if Fore
                                else f"        Skipped outcome: missing player or line: {outcome}"
                            )
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
                            print(
                                Fore.YELLOW
                                + f"        Skipped: missing price_over or price_under for {props}"
                                if Fore
                                else f"        Skipped: missing price_over or price_under for {props}"
                            )
                            continue
                        features = {
                            'line': props['line'],
                            'price_over': props['price_over'],
                            'price_under': props['price_under'],
                        }
                        prob = predict_pitcher_ks_over_probability(model_path, features)
                        prob_str = (
                            Fore.GREEN + f"{prob*100:.1f}%"
                            if Fore and prob is not None and prob > 0.6
                            else (
                                Fore.RED + f"{prob*100:.1f}%"
                                if Fore and prob is not None and prob < 0.4
                                else f"{prob*100:.1f}%" if prob is not None else "N/A"
                            )
                        )
                        print(
                            f"        EVAL: {props['player']} line={props['line']} over={props['price_over']} under={props['price_under']} prob={prob_str}"
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
    print(
        Fore.CYAN + f"DEBUG: Total evaluated props: {len(results)}"
        if Fore
        else f"DEBUG: Total evaluated props: {len(results)}"
    )
    return results


def print_projections_table(projections: list) -> None:
    if not projections:
        print(
            Fore.YELLOW + "No projection data available." if Fore else "No projection data available."
        )
        return

    headers = ["GAME", "BOOK", "PLAYER", "LINE", "OVER", "UNDER", "P(OVER)"]
    table = []
    for row in projections:
        prob = row.get("projected_over_probability")
        prob_val = f"{prob*100:.1f}%" if prob is not None else "N/A"
        # Colorize probability
        if Fore and prob is not None:
            if prob > 0.6:
                prob_str = Fore.GREEN + prob_val + Style.RESET_ALL
            elif prob < 0.4:
                prob_str = Fore.RED + prob_val + Style.RESET_ALL
            else:
                prob_str = Fore.YELLOW + prob_val + Style.RESET_ALL
        else:
            prob_str = prob_val
        table.append([
            row.get("game", ""),
            row.get("bookmaker", ""),
            row.get("player", ""),
            row.get("line", ""),
            row.get("price_over", ""),
            row.get("price_under", ""),
            prob_str,
        ])
    if tabulate:
        print(tabulate(table, headers, tablefmt="fancy_grid"))
    else:
        # Fallback: Plain text
        col_widths = [max(len(str(x)) for x in [h] + [row[i] for row in table]) for i, h in enumerate(headers)]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        for row in table:
            print(" | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))


def evaluate_h2h_all_tomorrow(
    sport_key: str,
    model_path: str,
    regions: str = "us",
) -> list:
    """Evaluate head-to-head win probability for all games in today's window."""

    from ml import predict_h2h_probability

    events = fetch_events(sport_key, regions=regions)
    results: list[dict] = []

    print(f"DEBUG: {len(events)} events returned by API")
    for event in events:
        commence = event.get("commence_time", "")
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        print(f"\nEVENT: {event_id} | {away} at {home} | {commence}")

        try:
            commence_dt = datetime.strptime(commence, "%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            print(f"  Skipped: invalid commence_time format {commence} ({e})")
            continue

        today = datetime.utcnow()
        start_dt = datetime(today.year, today.month, today.day, 16, 0, 0)
        end_dt = start_dt + timedelta(hours=14)

        if not (start_dt <= commence_dt < end_dt):
            print(
                f"  Skipped: commence_time {commence_dt} not in extended window {start_dt} to {end_dt}"
            )
            continue

        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets="h2h",
            regions=regions,
            player_props=False,
        )
        print(f"  Raw odds for event {event_id}:")
        print(json.dumps(game_odds, indent=2))

        if isinstance(game_odds, dict) and "bookmakers" in game_odds:
            game_odds = [game_odds]
        elif not isinstance(game_odds, list):
            print(f"  Skipped: unexpected odds format: {type(game_odds)} {game_odds}")
            continue

        for game in game_odds:
            if not isinstance(game, dict):
                print(f"  Skipped: game is not a dict: {game}")
                continue
            if not game.get("bookmakers"):
                print(f"  Skipped: no bookmakers in game {game.get('id')} for this event")
                continue
            print(
                f"  Bookmakers found: {[b.get('title') or b.get('key') for b in game.get('bookmakers', [])]}"
            )

            for book in game.get("bookmakers", []):
                book_name = book.get("title") or book.get("key")
                print(f"    Bookmaker: {book_name}")
                if not book.get("markets"):
                    print("      Skipped: no markets in this bookmaker")
                    continue
                for market in book.get("markets", []):
                    print(
                        f"      Market key: {market.get('key')}, desc: {market.get('description')}"
                    )
                    if market.get("key") != "h2h":
                        print("        Skipped: not a h2h market")
                        continue
                    if not market.get("outcomes"):
                        print("        Skipped: no outcomes in market")
                        continue
                    if len(market.get("outcomes", [])) != 2:
                        print("        Skipped: h2h market does not have exactly 2 outcomes")
                        continue
                    outcome1, outcome2 = market["outcomes"]
                    team1 = outcome1.get("name")
                    team2 = outcome2.get("name")
                    price1 = outcome1.get("price")
                    price2 = outcome2.get("price")
                    if team1 is None or team2 is None or price1 is None or price2 is None:
                        print("        Skipped: missing team name or price")
                        continue
                    prob = predict_h2h_probability(model_path, price1, price2)
                    print(
                        f"        EVAL: {team1}({price1}) vs {team2}({price2}) prob(team1 win)={prob}"
                    )
                    results.append(
                        {
                            "game": f"{team1} vs {team2}",
                            "bookmaker": book_name,
                            "team1": team1,
                            "team2": team2,
                            "price1": price1,
                            "price2": price2,
                            "event_id": event_id,
                            "projected_team1_win_probability": prob,
                        }
                    )
    print(f"DEBUG: Total evaluated h2h: {len(results)}")
    return results


def print_h2h_projections_table(projections: list) -> None:
    """Display a simple table for h2h projections."""

    if not projections:
        print("No projection data available.")
        return

    headers = [
        "GAME",
        "BOOK",
        "TEAM1",
        "TEAM2",
        "PRICE1",
        "PRICE2",
        "P(TEAM1 WIN)",
    ]

    def col_width(key: str, minimum: int) -> int:
        return max(minimum, max(len(str(row.get(key, ""))) for row in projections))

    widths = {
        "GAME": col_width("game", 10),
        "BOOK": col_width("bookmaker", 6),
        "TEAM1": col_width("team1", 8),
        "TEAM2": col_width("team2", 8),
        "PRICE1": col_width("price1", 6),
        "PRICE2": col_width("price2", 6),
        "P(TEAM1 WIN)": 13,
    }

    header_line = " ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in projections:
        prob = row.get("projected_team1_win_probability")
        prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"
        values = [
            row.get("game", ""),
            row.get("bookmaker", ""),
            row.get("team1", ""),
            row.get("team2", ""),
            row.get("price1", ""),
            row.get("price2", ""),
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
        print(Fore.YELLOW + "No upcoming events found." if Fore else "No upcoming events found.")
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
        description='Display projected batter strikeout props or head-to-head win probabilities for tomorrow (autofetch event IDs).'
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
    parser.add_argument('--h2h', action='store_true', help='Evaluate head-to-head win probabilities')
    parser.add_argument(
        '--list-events',
        action='store_true',
        help='List upcoming events for the given sport and exit'
    )
    args = parser.parse_args()

    if args.event_odds:
        if not args.event_id:
            print(
                Fore.RED + '--event-id is required with --event-odds'
                if Fore
                else '--event-id is required with --event-odds'
            )
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
            print(
                Fore.YELLOW + 'No upcoming events found.'
                if Fore
                else 'No upcoming events found.'
            )
            return
        for event in events:
            commence = event.get('commence_time', 'N/A')
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            event_id = event.get('id', '')
            print(f"{commence} - {away} at {home} ({event_id})")
        return

    if args.h2h:
        if args.model == 'pitcher_ks_classifier.pkl':
            args.model = 'h2h_classifier.pkl'
        projections = evaluate_h2h_all_tomorrow(
            args.sport,
            args.model,
            regions=args.regions,
        )
        print_h2h_projections_table(projections)
    else:
        projections = evaluate_batter_strikeouts_all_tomorrow(
            args.sport,
            args.model,
            regions=args.regions,
        )
        print_projections_table(projections)


if __name__ == '__main__':
    main()
