import os
import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import numpy as np
import pickle

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

# Import here to avoid circular imports
from ml import H2H_MODEL_PATH, predict_h2h_probability


def create_simple_fallback_model(model_path):
    """Create a simple model that directly converts odds to probabilities without ML"""
    print(f"{Fore.YELLOW}No trained model found at {model_path}" if Fore else f"No trained model found at {model_path}")
    print("Creating a simple fallback model based on American odds conversion...")

    class SimpleOddsModel:
        def predict_proba(self, X):
            price1 = X['price1'].values[0]
            if price1 > 0:
                prob = 100 / (price1 + 100)
            else:
                prob = abs(price1) / (abs(price1) + 100)
            return np.array([[1 - prob, prob]])

    model = SimpleOddsModel()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"{Fore.GREEN}Simple fallback model created at {model_path}" if Fore else f"Simple fallback model created at {model_path}")
    print("Note: This is a very basic model based on odds conversion. For better results, train with real data:")
    print("python ml.py --sport baseball_mlb --start-date 2023-05-01 --end-date 2023-06-01 --verbose --once")

    return model


def ensure_model_exists(model_path):
    """Ensure a model file exists, creating a fallback if needed."""
    path = Path(model_path)
    if not path.exists():
        create_simple_fallback_model(path)
    return str(path)


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
    markets: str = "h2h",
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
    markets: str = "h2h",
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


def evaluate_h2h_all_tomorrow(
    sport_key: str,
    model_path: str,
    regions: str = "us",
    verbose: bool = False,
) -> list:
    """Evaluate head-to-head win probability for all games in today's window."""
    
    # Ensure model exists
    model_path = ensure_model_exists(model_path)
    
    events = fetch_events(sport_key, regions=regions)
    results = []

    if verbose or True:
        print(f"DEBUG: {len(events)} events returned by API")
    
    now = datetime.utcnow()
    testing_mode = now.year >= 2025

    for event in events:
        commence = event.get("commence_time", "")
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        
        if verbose:
            print(f"\nEVENT: {event_id} | {away} at {home} | {commence}")

        try:
            commence_dt = datetime.strptime(commence, "%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            if verbose:
                print(f"  Skipped: invalid commence_time format {commence} ({e})")
            continue

        if not testing_mode:
            today = datetime.utcnow()
            start_dt = datetime(today.year, today.month, today.day, 16, 0, 0)
            end_dt = start_dt + timedelta(hours=14)

            if not (start_dt <= commence_dt < end_dt):
                if verbose:
                    print(f"  Skipped: commence_time {commence_dt} not in window {start_dt} to {end_dt}")
                continue

        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets="h2h",
            regions=regions,
            player_props=False,
        )
        
        if verbose:
            print(f"  Raw odds for event {event_id}:")
            print(json.dumps(game_odds, indent=2))

        if isinstance(game_odds, dict) and "bookmakers" in game_odds:
            game_odds = {"bookmakers": game_odds["bookmakers"]}
        elif not isinstance(game_odds, dict):
            if verbose:
                print(f"  Skipped: unexpected odds format: {type(game_odds)}")
            continue

        bookmakers = game_odds.get("bookmakers", [])
        if not bookmakers:
            if verbose:
                print(f"  Skipped: no bookmakers in game {event_id}")
            continue

        if verbose:
            print(f"  Bookmakers found: {[b.get('title') or b.get('key') for b in bookmakers]}")

        for book in bookmakers:
                book_name = book.get("title") or book.get("key")
                if verbose:
                    print(f"    Bookmaker: {book_name}")
                
                if not book.get("markets"):
                    if verbose:
                        print("      Skipped: no markets in this bookmaker")
                    continue
                
                for market in book.get("markets", []):
                    if verbose:
                        print(f"      Market key: {market.get('key')}, desc: {market.get('description')}")
                    
                    if market.get("key") != "h2h":
                        if verbose:
                            print("        Skipped: not a h2h market")
                        continue
                    
                    if not market.get("outcomes"):
                        if verbose:
                            print("        Skipped: no outcomes in market")
                        continue
                    
                    if len(market.get("outcomes", [])) != 2:
                        if verbose:
                            print("        Skipped: h2h market does not have exactly 2 outcomes")
                        continue
                    
                    outcome1, outcome2 = market["outcomes"]
                    team1 = outcome1.get("name")
                    team2 = outcome2.get("name")
                    price1 = outcome1.get("price")
                    price2 = outcome2.get("price")
                    
                    if team1 is None or team2 is None or price1 is None or price2 is None:
                        if verbose:
                            print("        Skipped: missing team name or price")
                        continue
                    
                    prob = predict_h2h_probability(model_path, price1, price2)
                    
                    if verbose:
                        print(f"        EVAL: {team1}({price1}) vs {team2}({price2}) prob(team1 win)={prob}")
                    
                    results.append({
                        "game": f"{team1} vs {team2}",
                        "bookmaker": book_name,
                        "team1": team1,
                        "team2": team2,
                        "price1": price1,
                        "price2": price2,
                        "event_id": event_id,
                        "projected_team1_win_probability": prob,
                    })
    
    if verbose:
        print(f"DEBUG: Total evaluated h2h: {len(results)}")
    
    return results


def print_h2h_projections_table(projections: list) -> None:
    """Display a simple table for h2h projections."""

    if not projections:
        print("No projection data available.")
        return

    if tabulate is not None:
        table_data = []
        for row in projections:
            prob = row.get("projected_team1_win_probability")
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

            price1 = row.get("price1", 0)
            price2 = row.get("price2", 0)
            price1_str = f"+{price1}" if price1 > 0 else f"{price1}"
            price2_str = f"+{price2}" if price2 > 0 else f"{price2}"

            table_data.append([
                row.get("team1", ""),
                price1_str,
                row.get("team2", ""),
                price2_str,
                prob_str,
                row.get("bookmaker", "")
            ])

        print(tabulate(
            table_data,
            headers=["Team 1", "Odds", "Team 2", "Odds", "Win Prob", "Book"],
            tablefmt="pretty"
        ))
    else:
        headers = [
            "TEAM1",
            "PRICE1",
            "TEAM2",
            "PRICE2",
            "P(WIN)",
            "BOOK",
        ]

        def col_width(key: str, minimum: int) -> int:
            return max(minimum, max(len(str(row.get(key, ""))) for row in projections))

        widths = {
            "TEAM1": col_width("team1", 10),
            "PRICE1": col_width("price1", 6),
            "TEAM2": col_width("team2", 10),
            "PRICE2": col_width("price2", 6),
            "P(WIN)": 8,
            "BOOK": col_width("bookmaker", 8),
        }

        header_line = " ".join(h.ljust(widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))

        for row in projections:
            prob = row.get("projected_team1_win_probability")
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

            price1 = row.get("price1", 0)
            price2 = row.get("price2", 0)
            price1_str = f"+{price1}" if price1 > 0 else f"{price1}"
            price2_str = f"+{price2}" if price2 > 0 else f"{price2}"

            values = [
                row.get("team1", ""),
                price1_str,
                row.get("team2", ""),
                price2_str,
                prob_str,
                row.get("bookmaker", ""),
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

    market_keys = set()
    
    for event in events[:1]:  # Just look at the first event to save API calls
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
        
        if isinstance(game_odds, dict) and "bookmakers" in game_odds:
            for book in game_odds.get("bookmakers", []):
                for market in book.get("markets", []):
                    market_key = market.get("key")
                    market_desc = market.get("description") or ""
                    if market_key:
                        market_keys.add((market_key, market_desc))
    
    print("\n=== Available Market Keys ===")
    for key, desc in sorted(market_keys):
        print(f"Market key: {key}  {f'- {desc}' if desc else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Display projected head-to-head win probabilities for tomorrow (autofetch event IDs).'
    )
    parser.add_argument('--sport', default='baseball_mlb', help='Sport key')
    parser.add_argument('--regions', default='us', help='Comma separated regions (default: us)')
    parser.add_argument('--model', default=str(H2H_MODEL_PATH), help='Path to trained ML model')
    parser.add_argument('--markets', default='h2h', help='Comma separated market keys')
    parser.add_argument('--odds-format', default='american', help='Odds format')
    parser.add_argument('--date-format', default='iso', help='Date format')
    parser.add_argument('--event-id', help='Event ID for event odds request')
    parser.add_argument('--event-odds', action='store_true', help='Print raw odds for the given event ID and exit')
    parser.add_argument('--list-market-keys', action='store_true', help='List market keys for upcoming events and exit')
    parser.add_argument('--game-period-markets', help='Comma separated game period market keys to include')
    parser.add_argument('--no-player-props', action='store_true', help='Exclude player prop markets')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debugging output')
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

    # Main functionality - get projections
    projections = evaluate_h2h_all_tomorrow(
        args.sport,
        args.model,
        regions=args.regions,
        verbose=args.verbose,
    )
    
    print("\n===== PROJECTED WIN PROBABILITIES =====")
    print_h2h_projections_table(projections)


if __name__ == '__main__':
    main()
