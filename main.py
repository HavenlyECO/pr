import os
import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
import time
import argparse
import sys
import numpy as np
import pickle
from typing import Dict, TypedDict, Any

# For improved table and color output
try:
    from tabulate import tabulate
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print(
        "Please install tabulate and colorama for improved output: pip install -r requirements-dev.txt"
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
TEST_MODE = False
if not API_KEY:
    TEST_MODE = True
    print('THE_ODDS_API_KEY environment variable is not set; running in test mode.')

# Minimum edge required for a bet to be recommended
EDGE_THRESHOLD = 0.06

# Risk filter parameters
RISK_EDGE_THRESHOLD = 0.05
RISK_ODDS_LIMIT = -170

# Sportsbooks considered "soft" for pricing comparisons. These lines often
# lag sharper markets, creating short-lived arbitrage opportunities when they
# disagree.
SOFT_BOOKS = ("bovada", "mybookie", "betus")

# Import here to avoid circular imports
from ml import (
    H2H_MODEL_PATH,
    MONEYLINE_MODEL_PATH,
    train_h2h_classifier,
    train_dual_head_classifier,
    predict_h2h_probability,
    train_moneyline_classifier,
    predict_moneyline_probability,
    american_odds_to_prob,
    american_odds_to_payout,
    MARKET_MAKER_MIRROR_MODEL_PATH,
    train_market_maker_mirror_model,
    market_maker_mirror_score,
    extract_advanced_ml_features,
    extract_market_signals,
)
import ml
from bankroll import calculate_bet_size
from bet_logger import log_bets
from scores import fetch_scores, append_scores_history, SCORES_HISTORY_FILE

# Dictionary key constants used throughout this module
K_GAME = "game"
K_BOOKMAKER = "bookmaker"
K_TEAM1 = "team1"
K_TEAM2 = "team2"
K_PRICE1 = "price1"
K_PRICE2 = "price2"
K_IMPLIED_WIN = "implied_team1_win_probability"
K_EVENT_ID = "event_id"
K_PROJECTED_WIN = "projected_team1_win_probability"
K_EDGE = "edge"
K_PAYOUT = "payout"
K_EXPECTED_VALUE = "expected_value"
K_TICKET_PCT_TEAM1 = "ticket_percent_team1"
K_TICKET_PCT_TEAM2 = "ticket_percent_team2"
K_PUBLIC_FADE = "public_fade"
K_STALE_FLAG = "stale_line_flag"
K_MARKET_DISAGREEMENT_SCORE = "market_disagreement_score"
K_SOFT_BOOK_SPREAD = "soft_book_spread"
K_MULTI_BOOK_EDGE_SCORE = "multi_book_edge_score"
K_MARKET_MAKER_MIRROR_SCORE = "market_maker_mirror_score"
K_RISK_WEIGHT = "risk_weight"
K_RISK_BLOCK_FLAG = "risk_block_flag"
K_WEIGHTED_EDGE = "weighted_edge"
K_WEIGHTED_EV = "weighted_expected_value"


class ProjectionRow(TypedDict, total=False):
    game: str
    bookmaker: str
    team1: str
    team2: str
    price1: int
    price2: int
    implied_team1_win_probability: float
    event_id: str
    projected_team1_win_probability: float
    edge: float
    payout: float
    expected_value: float
    ticket_percent_team1: float
    ticket_percent_team2: float
    public_fade: bool
    stale_line_flag: bool
    market_disagreement_score: float
    soft_book_spread: float | None
    multi_book_edge_score: float | None
    market_maker_mirror_score: float | None
    risk_weight: float
    risk_block_flag: bool
    weighted_edge: float
    weighted_expected_value: float

# Track last seen moneyline for stale line detection
_STALE_HISTORY: dict[str, dict[str, tuple[float, datetime]]] = {}


def _check_stale_line(
    event_id: str,
    bookmaker: str,
    price: float,
    *,
    threshold: float = 10,
    stale_seconds: int = 120,
) -> bool:
    """Return ``True`` when ``price`` is likely stale compared to other books.

    A line is marked stale if it hasn't moved for ``stale_seconds`` and the
    difference from the average of all other books is ``threshold`` or more.

    Example
    -------
    >>> _check_stale_line("e1", "bookA", 100)
    False
    >>> _check_stale_line("e1", "bookB", 110)
    False
    # assume two minutes pass with bookA unchanged while bookB moves
    >>> _check_stale_line("e1", "bookA", 100)
    True
    """
    now = datetime.utcnow()
    hist = _STALE_HISTORY.setdefault(event_id, {})
    last_price, last_time = hist.get(bookmaker, (None, None))
    others = [p for b, (p, _) in hist.items() if b != bookmaker]
    flagged = False
    if others:
        avg_others = sum(others) / len(others)
        diff = abs(price - avg_others)
        if diff >= threshold and last_price == price and last_time and (now - last_time).total_seconds() >= stale_seconds:
            flagged = True
    hist[bookmaker] = (price, now)
    _STALE_HISTORY[event_id] = hist
    return flagged


def risk_filter(edge: float | None, odds: float | None) -> float:
    """Return a weight of 0 when ``edge`` or ``odds`` fail risk checks."""

    if edge is None or odds is None:
        return 1.0
    if edge < RISK_EDGE_THRESHOLD or odds <= RISK_ODDS_LIMIT:
        return 0.0
    return 1.0


def ensure_model_exists(model_path):
    """Return the path if a trained model exists or raise an error."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Train one with 'python3 main.py train_classifier'"
        )
    return str(path)


def tomorrow_iso() -> str:
    return (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')


def build_events_url(
    sport_key: str,
    regions: str = "us",
    markets: str | None = None,
    bookmakers: str | None = None,
) -> str:
    params = {"apiKey": API_KEY, "regions": regions}
    if markets:
        params["markets"] = markets
    if bookmakers:
        params["bookmakers"] = bookmakers
    query = urllib.parse.urlencode(params)
    return f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?{query}"


def fetch_events(
    sport_key: str,
    regions: str = "us",
    markets: str | None = None,
    bookmakers: str | None = None,
) -> list:
    if TEST_MODE:
        print("Test mode active: fetch_events returning empty list")
        return []
    url = build_events_url(
        sport_key,
        regions=regions,
        markets=markets,
        bookmakers=bookmakers,
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
    if TEST_MODE:
        print("Test mode active: fetch_event_odds returning empty list")
        return []
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


def fetch_consensus_ticket_percentages(
    sport_key: str, regions: str = "us"
) -> Dict[str, Dict[str, float]]:
    """Return mapping of event id -> team -> ticket percentage."""
    if TEST_MODE:
        print(
            "Test mode active: fetch_consensus_ticket_percentages returning empty mapping"
        )
        return {}

    url = build_events_url(
        sport_key,
        regions=regions,
        markets="h2h",
        bookmakers="consensus",
    )

    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(
            Fore.RED + f"HTTPError fetching consensus data: {e.code} {e.reason}" if Fore else f"HTTPError fetching consensus data: {e.code} {e.reason}"
        )
        return {}
    except Exception as e:
        print(
            Fore.RED + f"Error fetching consensus data: {e}" if Fore else f"Error fetching consensus data: {e}"
        )
        return {}

    sentiments: Dict[str, Dict[str, float]] = {}
    if isinstance(data, list):
        for event in data:
            event_id = event.get("id")
            if not event_id:
                continue
            for book in event.get("bookmakers", []):
                if book.get("key") != "consensus":
                    continue
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    percentages: Dict[str, float] = {}
                    for outcome in market.get("outcomes", []):
                        team = outcome.get("name")
                        pct = outcome.get("ticket_percentage")
                        if team is not None and pct is not None:
                            percentages[team] = pct
                    if percentages:
                        sentiments[event_id] = percentages
                break
    return sentiments


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
    ticket_sentiments = fetch_consensus_ticket_percentages(sport_key, regions=regions)
    results = []

    if verbose or True:
        print(f"DEBUG: {len(events)} events returned by API")
    
    now = datetime.utcnow()
    testing_mode = TEST_MODE or now.year >= 2025

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

        if verbose:
            msg = f"[FETCHING] Odds for {away} at {home} ({event_id})..."
            print(msg, end="", flush=True)
        game_odds = fetch_event_odds(
            sport_key,
            event_id,
            markets="h2h",
            regions=regions,
            player_props=False,
        )

        if verbose:
            print(" done.")
        
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
            print(
                f"  Bookmakers found: {[b.get('title') or b.get('key') for b in bookmakers]}"
            )

        event_rows: list[dict] = []
        implied_probs: list[float] = []
        soft_implied_probs: list[float] = []

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
                    print(
                        f"      Market key: {market.get('key')}, desc: {market.get('description')}"
                    )

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
                        print(
                            "        Skipped: h2h market does not have exactly 2 outcomes"
                        )
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

                implied = american_odds_to_prob(price1)
                edge = prob - implied
                payout = american_odds_to_payout(price1)
                ev = edge * payout

                book_key = (book_name or "").lower()
                if any(sb in book_key for sb in SOFT_BOOKS):
                    soft_implied_probs.append(implied)

                sentiment = ticket_sentiments.get(event_id)
                team1_pct = team2_pct = None
                fade = False
                if sentiment:
                    team1_pct = sentiment.get(team1)
                    team2_pct = sentiment.get(team2)
                    if team1_pct is not None and team1_pct >= 70 and edge < 0:
                        fade = True

                stale = _check_stale_line(event_id, book_name, price1)

                if verbose:
                    print(
                        f"        EVAL: {team1}({price1}) vs {team2}({price2}) "
                        f"prob={prob:.3f} implied={implied:.3f} edge={edge:.3f} "
                        f"payout={payout:.3f} EV={ev:.4f}"
                    )

                row = {
                    K_GAME: f"{team1} vs {team2}",
                    K_BOOKMAKER: book_name,
                    K_TEAM1: team1,
                    K_TEAM2: team2,
                    K_PRICE1: price1,
                    K_PRICE2: price2,
                    K_IMPLIED_WIN: implied,
                    K_EVENT_ID: event_id,
                    K_PROJECTED_WIN: prob,
                    K_EDGE: edge,
                    K_PAYOUT: payout,
                    K_EXPECTED_VALUE: ev,
                    K_TICKET_PCT_TEAM1: team1_pct,
                    K_TICKET_PCT_TEAM2: team2_pct,
                    K_PUBLIC_FADE: fade,
                    K_STALE_FLAG: stale,
                }

                # Add social media data if available
                if hasattr(ml, "llm_sharp_social_score") and team1:
                    try:
                        row["reddit_sentiment"] = ml.llm_sharp_social_score(team1)
                        row["hype_trend"] = ml.llm_hype_trend_social_score(team1)
                        row["lineup_risk"] = ml.llm_lineup_risk_social_score(team1)
                    except Exception as e:
                        print(f"Social feature error: {e}")

                event_rows.append(row)
                implied_probs.append(implied)

        if not event_rows:
            continue

        diff = max(implied_probs) - min(implied_probs)

        soft_spread = None
        multi_book_edge = None
        if soft_implied_probs:
            high = max(soft_implied_probs)
            low = min(soft_implied_probs)
            soft_spread = high - low
            multi_book_edge = (high + low) / 2

        for row in event_rows:
            row[K_MARKET_DISAGREEMENT_SCORE] = diff
            row[K_SOFT_BOOK_SPREAD] = soft_spread
            row[K_MULTI_BOOK_EDGE_SCORE] = multi_book_edge
            mm_features = {
                "opening_odds": row.get(K_PRICE1),
                "handle_percent": row.get(K_TICKET_PCT_TEAM1, 0.0),
                "ticket_percent": row.get(K_TICKET_PCT_TEAM1, 0.0),
                "volatility": diff * 100,
            }
            if Path(MARKET_MAKER_MIRROR_MODEL_PATH).exists():
                row[K_MARKET_MAKER_MIRROR_SCORE] = market_maker_mirror_score(
                    str(MARKET_MAKER_MIRROR_MODEL_PATH),
                    mm_features,
                    row.get(K_PRICE1),
                )
            else:
                row[K_MARKET_MAKER_MIRROR_SCORE] = None

            # Add advanced machine learning features
            team1 = row.get(K_TEAM1)
            team2 = row.get(K_TEAM2)
            if Path(MONEYLINE_MODEL_PATH).exists():
                adv = extract_advanced_ml_features(
                    model_path=str(MONEYLINE_MODEL_PATH),
                    price1=row.get(K_PRICE1),
                    price2=row.get(K_PRICE2),
                    team1=team1,
                    team2=team2,
                )
                row.update(adv)

            if Path(MARKET_MAKER_MIRROR_MODEL_PATH).exists():
                sig = extract_market_signals(
                    model_path=str(MARKET_MAKER_MIRROR_MODEL_PATH),
                    price1=row.get(K_PRICE1),
                    ticket_percent=row.get(K_TICKET_PCT_TEAM1, 0.0),
                )
                row.update(sig)
            weight = 1 + diff + (soft_spread or 0)
            if row.get(K_STALE_FLAG):
                weight *= 1.1
            r_weight = risk_filter(row.get(K_EDGE), row.get(K_PRICE1))
            row[K_RISK_WEIGHT] = r_weight
            row[K_RISK_BLOCK_FLAG] = r_weight == 0.0
            weight *= r_weight
            row[K_WEIGHTED_EDGE] = row[K_EDGE] * weight
            row[K_WEIGHTED_EV] = row[K_EXPECTED_VALUE] * weight
            results.append(row)
    
    if verbose:
        print(f"DEBUG: Total evaluated h2h: {len(results)}")
    
    return results


def print_h2h_projections_table(projections: list) -> None:
    """Display a visually appealing table for h2h projections."""

    if not projections:
        print("No projection data available.")
        return


    if tabulate is not None:
        # Create table data with better formatting
        table_data: list[list[str]] = []
        use_color = Fore is not None

        for row in projections:
            # Add recommendation indicator
            rec = ""
            edge = row.get(K_EDGE)
            if edge is not None and edge > EDGE_THRESHOLD and not row.get(K_RISK_BLOCK_FLAG):
                rec = f"{Fore.GREEN}★{Style.RESET_ALL}" if use_color else "★"

            # Format team names - highlight home team
            team1 = row.get(K_TEAM1, "")
            team2 = row.get(K_TEAM2, "")
            if use_color:
                team1 = f"{Fore.CYAN}{team1}{Style.RESET_ALL}"

            # Format probabilities
            prob = row.get(K_PROJECTED_WIN)
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

            # Format edges with colors based on value
            edge_color = ""
            edge_reset = ""
            if use_color and edge is not None:
                if edge > EDGE_THRESHOLD:
                    edge_color = Fore.GREEN
                elif edge < 0:
                    edge_color = Fore.RED
                edge_reset = Style.RESET_ALL
            edge_str = f"{edge_color}{edge*100:+.1f}%{edge_reset}" if edge is not None else "N/A"

            # Format weighted edge
            w_edge = row.get(K_WEIGHTED_EDGE)
            w_edge_str = f"{w_edge*100:+.1f}%" if w_edge is not None else "N/A"

            # Format EV
            ev = row.get(K_EXPECTED_VALUE)
            ev_str = f"{ev:+.3f}" if ev is not None else "N/A"

            # Format odds
            price1 = row.get(K_PRICE1, 0)
            price2 = row.get(K_PRICE2, 0)
            price1_str = f"+{price1}" if price1 > 0 else f"{price1}"
            price2_str = f"+{price2}" if price2 > 0 else f"{price2}"

            # Format ML confidence
            ml_conf = row.get("ml_confidence")
            ml_conf_str = f"{ml_conf:.3f}" if ml_conf is not None else "N/A"

            # Add row to table
            table_data.append([
                rec,
                team1,
                price1_str,
                team2,
                price2_str,
                prob_str,
                edge_str,
                ev_str,
                row.get(K_BOOKMAKER, "")[:12],
            ])

        # Print the nicely formatted table
        headers = ["Rec", "Team", "Odds", "Opponent", "Odds", "Win%", "Edge", "EV", "Book"]
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

        # Add legend if any recommendations exist
        if any(row[0] for row in table_data):
            star = f"{Fore.GREEN}★{Style.RESET_ALL}" if use_color else "★"
            print(f"\n{star} = Recommended bet (Edge > {EDGE_THRESHOLD*100:.1f}%)")
    else:
        # Fallback formatter for environments without tabulate
        headers = ["REC", "TEAM", "ODDS", "OPP", "ODDS", "WIN%", "EDGE", "EV", "BOOK"]

        def col_width(key: str, minimum: int) -> int:
            return max(minimum, max(len(str(row.get(key, ""))) for row in projections))

        widths = {
            "REC": 3,
            "TEAM": col_width(K_TEAM1, 15),
            "ODDS": 6,
            "OPP": col_width(K_TEAM2, 15),
            "ODDS2": 6,
            "WIN%": 6,
            "EDGE": 7,
            "EV": 6,
            "BOOK": 12,
        }

        # Print header
        header_line = " ".join(h.ljust(widths[h if h != "ODDS2" else "ODDS"]) for h in headers)
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in projections:
            edge = row.get(K_EDGE)
            rec = "★" if edge is not None and edge > EDGE_THRESHOLD and not row.get(K_RISK_BLOCK_FLAG) else " "

            prob = row.get(K_PROJECTED_WIN)
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

            edge_str = f"{edge*100:+.1f}%" if edge is not None else "N/A"

            ev = row.get(K_EXPECTED_VALUE)
            ev_str = f"{ev:+.3f}" if ev is not None else "N/A"

            price1 = row.get(K_PRICE1, 0)
            price2 = row.get(K_PRICE2, 0)
            price1_str = f"+{price1}" if price1 > 0 else f"{price1}"
            price2_str = f"+{price2}" if price2 > 0 else f"{price2}"

            values = [
                rec,
                row.get(K_TEAM1, ""),
                price1_str,
                row.get(K_TEAM2, ""),
                price2_str,
                prob_str,
                edge_str,
                ev_str,
                row.get(K_BOOKMAKER, "")[:12],
            ]

            print(" ".join(str(v).ljust(widths[h if i != 4 else "ODDS2"]) for i, (v, h) in enumerate(zip(values, headers))))

        # Print legend when recommendations exist
        has_recommendations = any(
            row.get(K_EDGE, 0) > EDGE_THRESHOLD and not row.get(K_RISK_BLOCK_FLAG, False)
            for row in projections
        )
        if has_recommendations:
            print(f"\n★ = Recommended bet (Edge > {EDGE_THRESHOLD*100:.1f}%)")


def log_bet_recommendations(
    projections: list,
    *,
    threshold: float = EDGE_THRESHOLD,
    bankroll: float | None = None,
    kelly_fraction: float = 1.0,
    log_file: str = "bet_recommendations.log",
) -> None:
    """Append bet recommendations to ``log_file``.

    This produces a lightweight, human-readable text log. Bets are only
    logged when their edge exceeds ``threshold`` (defaults to
    ``EDGE_THRESHOLD``). The edge is the model's predicted win probability minus
    the implied probability from the offered odds. When ``bankroll`` is
    supplied the recommended stake is calculated using ``calculate_bet_size``.

    For a structured JSONL record of the same bets use :func:`log_bets` which
    writes to ``bet_log.jsonl``.
    """

    lines: list[str] = []
    for row in projections:
        edge = row.get(K_WEIGHTED_EDGE, row.get(K_EDGE))
        if row.get(K_RISK_BLOCK_FLAG):
            continue
        prob = row.get(K_PROJECTED_WIN)
        if edge is None or prob is None or edge <= threshold:
            continue
        team = row.get(K_TEAM1, "")
        odds = row.get(K_PRICE1)
        bookmaker = row.get(K_BOOKMAKER, "")
        timestamp = datetime.utcnow().isoformat()
        line = (
            f"{timestamp} - {team} @ {odds} ({bookmaker}) "
            f"prob={prob:.3f} edge={edge:+.3f}"
        )
        if bankroll is not None:
            stake = calculate_bet_size(bankroll, prob, odds, fraction=kelly_fraction)
            line += f" stake={stake:.2f}"
        lines.append(line)

    if not lines:
        return

    with open(log_file, "a") as f:
        for line in lines:
            f.write(line + "\n")


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


def _parse_year_range(text: str) -> range:
    """Return ``range`` object from YEAR or YEAR-YEAR string."""
    if not text:
        return range(2018, datetime.utcnow().year)
    if "-" in text:
        start, end = text.split("-", 1)
    else:
        start = end = text
    start_i = int(start)
    end_i = int(end)
    if start_i > end_i:
        start_i, end_i = end_i, start_i
    return range(start_i, end_i + 1)


def _summarize_projections(projections: list[ProjectionRow]) -> list[ProjectionRow]:
    """Return the best row per event ordered by weighted EV descending."""

    best_rows: dict[str, ProjectionRow] = {}
    for row in projections:
        event_id = row.get(K_EVENT_ID)
        if not event_id:
            continue
        ev = row.get(K_WEIGHTED_EV, row.get(K_EXPECTED_VALUE) or 0.0)
        current = best_rows.get(event_id)
        current_ev = 0.0
        if current is not None:
            current_ev = current.get(K_WEIGHTED_EV, current.get(K_EXPECTED_VALUE) or 0.0)
        if current is None or ev > current_ev:
            best_rows[event_id] = row

    ordered = sorted(best_rows.values(), key=lambda r: r.get(K_WEIGHTED_EV, r.get(K_EXPECTED_VALUE) or 0.0), reverse=True)
    return ordered


def run_pipeline(
    *, sport: str = "baseball_mlb", regions: str = "us", model_path: str = str(H2H_MODEL_PATH), verbose: bool = False
) -> None:
    """Fetch live data, compute features, run predictions and print a dashboard."""

    projections = evaluate_h2h_all_tomorrow(
        sport,
        model_path,
        regions=regions,
        verbose=verbose,
    )

    if not projections:
        print("No data returned from API")
        return

    print("\n===== PROJECTED WIN PROBABILITIES =====")
    print_h2h_projections_table(projections)
    log_bet_recommendations(projections, threshold=EDGE_THRESHOLD)
    log_bets(projections, threshold=EDGE_THRESHOLD)

    dashboard_rows = _summarize_projections(projections)
    if dashboard_rows:
        print("\n===== TODAY'S TOP RECOMMENDATIONS =====")
        print_h2h_projections_table(dashboard_rows[:10])

    print(
        "Recommendations appended to bet_recommendations.log; "
        "detailed JSON records saved to bet_log.jsonl"
    )


def train_pipeline(*, years: str = "2018-2024", sport: str = "baseball_mlb", verbose: bool = False) -> None:
    """Build datasets and train all models for ``sport`` over ``years``."""

    year_range = _parse_year_range(years)

    import integrate_data

    integrate_data.YEARS_TO_PROCESS = year_range
    integrate_data.main()
    dataset_path = integrate_data.OUTPUT_FILE

    train_dual_head_classifier(dataset_path, verbose=verbose)

    start = min(year_range)
    end = max(year_range)
    train_h2h_classifier(
        sport,
        f"{start}-01-01",
        f"{end}-12-31",
        verbose=verbose,
    )
    print("Training complete.")


def train_classifier_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Train moneyline classifier")
    parser.add_argument("--dataset", required=True, help="CSV file with training data")
    parser.add_argument("--model-out", default=str(MONEYLINE_MODEL_PATH))
    parser.add_argument(
        "--features-type",
        choices=["pregame", "live", "dual"],
        default="pregame",
        help="Which feature set to use for training (or 'dual' for both)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--recent-half-life",
        type=float,
        help="Half-life in days for weighting recent games",
    )
    parser.add_argument(
        "--recency-multiplier",
        type=float,
        default=0.7,
        help="Weight to apply to recent stat columns when combining with season averages",
    )
    parser.add_argument(
        "--date-column",
        help="Column name containing game dates for recency weighting",
    )
    args = parser.parse_args(argv)

    if args.features_type == "dual":
        train_dual_head_classifier(
            args.dataset,
            model_out=args.model_out,
            verbose=args.verbose,
            recent_half_life=args.recent_half_life,
            date_column=args.date_column,
            recency_multiplier=args.recency_multiplier,
        )
    else:
        train_moneyline_classifier(
            args.dataset,
            model_out=args.model_out,
            features_type=args.features_type,
            verbose=args.verbose,
            recent_half_life=args.recent_half_life,
            date_column=args.date_column,
            recency_multiplier=args.recency_multiplier,
        )


def predict_classifier_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Predict with moneyline classifier")
    parser.add_argument("--model", default=str(MONEYLINE_MODEL_PATH))
    parser.add_argument("--features", required=True, help="JSON encoded feature mapping")
    args = parser.parse_args(argv)

    features = json.loads(args.features)
    prob = predict_moneyline_probability(args.model, features)
    print(f"Home team win probability: {prob:.3f}")


def continuous_train_classifier_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Continuously train h2h classifier")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default=str(H2H_MODEL_PATH))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    next_run = datetime.utcnow()
    start = datetime.fromisoformat(args.start_date)
    while True:
        if datetime.utcnow() >= next_run:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
            train_h2h_classifier(
                args.sport,
                start.strftime("%Y-%m-%d"),
                end_date,
                model_out=args.model_out,
                verbose=args.verbose,
            )
            try:
                scores = fetch_scores(args.sport, days_from=3)
                append_scores_history(scores)
            except Exception as exc:  # pragma: no cover - keep running on error
                print(f"Failed to fetch scores: {exc}")
            next_run = datetime.utcnow() + timedelta(hours=args.interval_hours)
        time.sleep(30)


def continuous_train_moneyline_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Continuously train moneyline or dual-head classifier"
    )
    parser.add_argument("--dataset", required=True, help="CSV file with training data")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--features-type", choices=["pregame", "live", "dual"], default="pregame")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default=str(MONEYLINE_MODEL_PATH))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recent-half-life", type=float)
    parser.add_argument("--date-column")
    parser.add_argument("--recency-multiplier", type=float, default=0.7)
    args = parser.parse_args(argv)

    next_run = datetime.utcnow()
    while True:
        if datetime.utcnow() >= next_run:
            if args.features_type == "dual":
                train_dual_head_classifier(
                    args.dataset,
                    model_out=args.model_out,
                    verbose=args.verbose,
                    recent_half_life=args.recent_half_life,
                    date_column=args.date_column,
                    recency_multiplier=args.recency_multiplier,
                )
            else:
                train_moneyline_classifier(
                    args.dataset,
                    model_out=args.model_out,
                    features_type=args.features_type,
                    verbose=args.verbose,
                    recent_half_life=args.recent_half_life,
                    date_column=args.date_column,
                    recency_multiplier=args.recency_multiplier,
                )
            try:
                scores = fetch_scores(args.sport, days_from=3)
                append_scores_history(scores)
            except Exception as exc:  # pragma: no cover - keep running on error
                print(f"Failed to fetch scores: {exc}")
            next_run = datetime.utcnow() + timedelta(hours=args.interval_hours)
        time.sleep(30)


def continuous_train_mirror_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Continuously train market maker mirror model"
    )
    parser.add_argument("--dataset", required=True, help="CSV file with training data")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--model-out", default=str(MARKET_MAKER_MIRROR_MODEL_PATH))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    next_run = datetime.utcnow()
    while True:
        if datetime.utcnow() >= next_run:
            train_market_maker_mirror_model(
                args.dataset,
                model_out=args.model_out,
                verbose=args.verbose,
            )
            try:
                scores = fetch_scores(args.sport, days_from=3)
                append_scores_history(scores)
            except Exception as exc:  # pragma: no cover - keep running on error
                print(f"Failed to fetch scores: {exc}")
            next_run = datetime.utcnow() + timedelta(hours=args.interval_hours)
        time.sleep(30)


def scores_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Fetch recent scores")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--days-from", type=int, default=0)
    parser.add_argument("--save-history", action="store_true")
    args = parser.parse_args(argv)

    scores = fetch_scores(args.sport, days_from=args.days_from)
    print(json.dumps(scores, indent=2))
    if args.save_history:
        append_scores_history(scores)
        print(f"Scores saved to {SCORES_HISTORY_FILE}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "train_classifier":
        train_classifier_cli(argv[1:])
        return
    if argv and argv[0] == "predict_classifier":
        predict_classifier_cli(argv[1:])
        return
    if argv and argv[0] == "continuous_train_classifier":
        continuous_train_classifier_cli(argv[1:])
        return
    if argv and argv[0] == "continuous_train_moneyline":
        continuous_train_moneyline_cli(argv[1:])
        return
    if argv and argv[0] == "continuous_train_mirror":
        continuous_train_mirror_cli(argv[1:])
        return
    if argv and argv[0] == "scores":
        scores_cli(argv[1:])
        return

    parser = argparse.ArgumentParser(
        description='Display projected head-to-head win probabilities for tomorrow (autofetch event IDs).'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', action='store_true', help='Execute full live pipeline and exit')
    group.add_argument('--train', action='store_true', help='Run end-to-end training pipeline and exit')
    parser.add_argument('--years', help='Year or YEAR-YEAR range for training')
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
    args, remaining = parser.parse_known_args(argv)

    if remaining:
        parser.error('Unrecognized arguments: ' + ' '.join(remaining))

    if args.game_period_markets and not (args.event_odds or args.list_market_keys):
        print(
            (Fore.YELLOW if Fore else '') +
            '--game-period-markets has no effect without --event-odds or --list-market-keys'
        )

    if args.run:
        try:
            run_pipeline(
                sport=args.sport,
                regions=args.regions,
                model_path=args.model,
                verbose=args.verbose,
            )
        except FileNotFoundError as exc:
            print((Fore.RED + str(exc)) if Fore else str(exc))
            return
        return
    if args.train:
        train_pipeline(years=args.years or '2018-2024', sport=args.sport, verbose=args.verbose)
        return

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
    try:
        projections = evaluate_h2h_all_tomorrow(
            args.sport,
            args.model,
            regions=args.regions,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print((Fore.RED + str(exc)) if Fore else str(exc))
        return
    
    print("\n===== PROJECTED WIN PROBABILITIES =====")
    print_h2h_projections_table(projections)
    log_bet_recommendations(projections, threshold=EDGE_THRESHOLD)
    # Also log detailed bet information for later ROI analysis
    log_bets(projections, threshold=EDGE_THRESHOLD)
    print(
        "Recommendations appended to bet_recommendations.log; "
        "detailed JSON records saved to bet_log.jsonl"
    )


if __name__ == '__main__':
    main()
