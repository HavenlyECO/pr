import json
import urllib.request
import urllib.parse
import time
import os
import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except Exception:
    rapidfuzz_fuzz = None
try:
    from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz
except Exception:
    fuzzywuzzy_fuzz = None

# --------------- CONFIG -----------------
UNDERDOG_GQL_URL = "https://api.underdogfantasy.com/beta/v4/over_under_lines"
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DB_PATH = os.getenv("LINE_HISTORY_DB", "line_history.db")
CHECK_INTERVAL = 60  # seconds between checks
# Base threshold used when a stat-specific value is not provided
DEFAULT_EDGE_THRESHOLD = 0.7  # minimum difference to alert

# Optional per-stat thresholds allowing edges to vary by market type. These
# values can be tuned based on historical volatility for each stat.
EDGE_THRESHOLD_BY_STAT = {
    "strikeouts": 0.5,  # often lower variance
    "total bases": 1.5,  # typically higher variance
}

# Backwards compatible constant name (used by find_value_props signature)
EDGE_THRESHOLD = DEFAULT_EDGE_THRESHOLD
PLAYER_MATCH_THRESHOLD = 80  # fuzzy name match ratio threshold

# Bookmaker keys to use for consensus lines
BOOKMAKER_KEYS = ["draftkings", "fanduel", "pointsbetus", "betmgm"]

# Map normalized Underdog stat types to TheOdds API market keys.
# Additional stats can be added here as needed.
STAT_KEY_MAP = {
    "hits": ["player_hits"],
    "home runs": ["player_home_runs"],
    "total bases": ["player_total_bases"],
    "rbis": ["player_rbis"],
    "strikeouts": ["player_strikeouts"],
    "runs": ["player_runs"],
    "walks": ["player_walks"],
    "singles": ["player_singles"],
    "doubles": ["player_doubles"],
    "triples": ["player_triples"],
}

ALL_MARKETS = ",".join(sorted({m for v in STAT_KEY_MAP.values() for m in v}))


def init_db():
    """Initialize the SQLite database for line history."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS line_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_id TEXT,
            player TEXT,
            stat TEXT,
            line REAL,
            ts INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def save_line_history(props):
    """Save the current Underdog lines to the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = int(time.time())
    for p in props:
        cur.execute(
            "INSERT INTO line_history (line_id, player, stat, line, ts) VALUES (?, ?, ?, ?, ?)",
            (p.get("id"), p.get("player"), p.get("stat"), p.get("line"), ts),
        )
    conn.commit()
    conn.close()


def get_line_history(line_id):
    """Return historical line values for a given line id."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, line FROM line_history WHERE line_id=? ORDER BY ts",
        (line_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def plot_line_history(line_id):
    """Plot line movement over time for a given line."""
    rows = get_line_history(line_id)
    if not rows:
        print(f"No history found for line id {line_id}")
        return
    import matplotlib.pyplot as plt

    times = [datetime.fromtimestamp(r[0]) for r in rows]
    lines = [r[1] for r in rows]
    plt.plot(times, lines, marker="o")
    plt.title(f"Line history for {line_id}")
    plt.xlabel("Time")
    plt.ylabel("Line")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return a fuzzy match ratio between two strings.

    Uses RapidFuzz if available, otherwise fuzzywuzzy or a basic SequenceMatcher
    fallback.
    """
    if rapidfuzz_fuzz:
        return rapidfuzz_fuzz.token_set_ratio(a, b)
    if fuzzywuzzy_fuzz:
        return fuzzywuzzy_fuzz.token_set_ratio(a, b)
    return SequenceMatcher(None, a, b).ratio() * 100


def _normalize_name(name: str) -> str:
    """Normalize player names for fuzzy comparison."""
    name = name.lower().replace('.', '').replace(',', '')
    for suffix in [" jr", " sr", " jr", " sr", " iii", " ii"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def _normalize_stat_name(stat: str) -> str:
    """Normalize stat type names for matching."""
    stat = stat.lower()
    stat = re.sub(r"[^a-z0-9]+", " ", stat)
    return stat.strip()


def get_market_keys(stat: str):
    """Return TheOdds market keys for a given Underdog stat type."""
    norm = _normalize_stat_name(stat)
    if norm in STAT_KEY_MAP:
        return STAT_KEY_MAP[norm]
    for key, markets in STAT_KEY_MAP.items():
        if key in norm:
            return markets
    return []


def player_match(name_a: str, name_b: str, threshold: int = PLAYER_MATCH_THRESHOLD) -> bool:
    """Return True if two player names match with a ratio above the threshold."""
    score = _fuzzy_ratio(_normalize_name(name_a), _normalize_name(name_b))
    return score >= threshold

# --------------- EV CALCULATIONS -----------------
def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_prob(odds: float) -> float:
    """Return implied probability from American odds."""
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_hold(over_odds: float, under_odds: float):
    """Return sportsbook hold and normalized probabilities for each side."""
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total = p_over + p_under
    if total == 0:
        return 0, 0, 0
    hold = total - 1
    return hold, p_over / total, p_under / total


def compute_ev(true_prob: float, odds: float) -> float:
    """Return expected value for a bet with given true probability and odds."""
    dec = american_to_decimal(odds)
    return true_prob * dec - 1


def fetch_player_over_probability(player_name: str, stat: str, line: float, season: str = None, games: int = 100):
    """Return probability of player going over a line using MLB Stats API.

    This function queries the public MLB Stats API to retrieve recent game logs
    for a player and calculates the proportion of games in which the specified
    stat went over the provided line. Theoddsapi is deliberately not used so
    historical performance comes from a different source.
    """
    try:
        search_url = (
            "https://statsapi.mlb.com/api/v1/people/search?name="
            + urllib.parse.quote(player_name)
        )
        with urllib.request.urlopen(search_url) as resp:
            search_data = json.loads(resp.read().decode())
        results = search_data.get("people") or []
        if not results:
            return None
        player_id = results[0]["id"]
        params = {
            "stats": "gameLog",
            "group": "hitting",
            "sportId": 1,
            "limit": games,
        }
        if season:
            params["season"] = season
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?"
            + urllib.parse.urlencode(params)
        )
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:  # network or parse error
        print("Historical data error:", exc)
        return None

    splits = (
        data.get("stats", [{}])[0]
        .get("splits", [])
    )
    if not splits:
        return None

    over = 0
    total = 0
    for g in splits:
        stat_val = g.get("stat", {}).get(stat)
        if stat_val is None:
            continue
        try:
            val = float(stat_val)
        except Exception:
            continue
        total += 1
        if val > line:
            over += 1
    if total == 0:
        return None
    return over / total

# --------------- FETCH UNDERDOG MLB PROPS -----------------
def fetch_underdog_props():
    params = {
        "sport": "mlb",
        "platform": "web"
    }
    query_params = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{UNDERDOG_GQL_URL}?{query_params}") as resp:
        if resp.status != 200:
            raise RuntimeError(f"Underdog API error: {resp.status}")
        data = json.loads(resp.read().decode())
    props = []
    for line in data.get("over_under_lines", []):
        prop = {
            "player": line.get("over_under", {}).get("title"),
            "stat": line.get("over_under", {}).get("stat_type"),
            "line": line.get("line_score"),
            "game": line.get("over_under", {}).get("game", {}).get("matchup"),
            "id": line.get("id"),
        }
        # Filter out props without required info (sometimes happens)
        if prop["player"] and prop["stat"] and prop["line"]:
            props.append(prop)
    return props

# --------------- FETCH CONSENSUS MLB PROPS -----------------
def fetch_consensus_props():
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us",
        "markets": ALL_MARKETS,
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKER_KEYS),
    }
    query_params = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query_params}") as resp:
        if resp.status != 200:
            raise RuntimeError(f"TheOdds API error: {resp.status}")
        return json.loads(resp.read().decode())

# --------------- CONSENSUS AGGREGATION -----------------
def aggregate_consensus_line(cons_props, market_keys, player_name):
    """Return the average line for a player across configured bookmakers."""
    lines = []
    books = []
    for game in cons_props:
        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") not in BOOKMAKER_KEYS:
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") not in market_keys:
                    continue
                for outcome in market.get("outcomes", []):
                    if player_match(player_name, outcome.get("name", "")):
                        line = outcome.get("point")
                        if line is not None:
                            try:
                                lines.append(float(line))
                                books.append(bookmaker.get("title", bookmaker.get("key")))
                            except Exception:
                                pass
    if lines:
        return sum(lines) / len(lines), sorted(set(books))
    return None, []


def aggregate_market_with_odds(cons_props, market_keys, player_name):
    """Aggregate line and odds data for a player across bookmakers."""
    lines = []
    over_prices = []
    under_prices = []
    books = []
    for game in cons_props:
        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") not in BOOKMAKER_KEYS:
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") not in market_keys:
                    continue
                for outcome in market.get("outcomes", []):
                    if player_match(player_name, outcome.get("name", "")):
                        line = outcome.get("point")
                        if line is not None:
                            try:
                                lines.append(float(line))
                            except Exception:
                                pass
                        price = outcome.get("price")
                        if price is not None:
                            try:
                                price_f = float(price)
                                name = outcome.get("name", "").lower()
                                if "over" in name:
                                    over_prices.append(price_f)
                                elif "under" in name:
                                    under_prices.append(price_f)
                            except Exception:
                                pass
                        books.append(bookmaker.get("title", bookmaker.get("key")))
    avg_line = sum(lines) / len(lines) if lines else None
    avg_over = sum(over_prices) / len(over_prices) if over_prices else None
    avg_under = sum(under_prices) / len(under_prices) if under_prices else None
    return avg_line, avg_over, avg_under, sorted(set(books))

# --------------- FIND VALUE PROPS -----------------
def find_value_props(ud_props, cons_props, threshold=EDGE_THRESHOLD):
    """Return props where the line difference exceeds the stat threshold."""
    value_props = []
    for u in ud_props:
        market_keys = get_market_keys(u["stat"])
        if not market_keys:
            continue
        consensus_line, books = aggregate_consensus_line(cons_props, market_keys, u["player"])
        if consensus_line is not None:
            try:
                diff = float(u["line"]) - float(consensus_line)
                # Determine edge threshold for this stat type
                stat_threshold = EDGE_THRESHOLD_BY_STAT.get(
                    _normalize_stat_name(u["stat"]), threshold
                )
                if abs(diff) >= stat_threshold:
                    value_props.append({
                        "player": u["player"],
                        "stat": u["stat"],
                        "underdog_line": u["line"],
                        "consensus_line": consensus_line,
                        "diff": diff,
                        "game": u["game"],
                        "book": ", ".join(books) if books else "consensus",
                    })
            except Exception as e:
                print("Parse error:", e)
    return value_props


def find_ev_props(ud_props, cons_props, games=100):
    """Return props with calculated expected value using historical data."""
    ev_props = []
    for u in ud_props:
        market_keys = get_market_keys(u["stat"])
        if not market_keys:
            continue
        line, over_price, under_price, books = aggregate_market_with_odds(
            cons_props, market_keys, u["player"]
        )
        if line is None or over_price is None or under_price is None:
            continue
        true_over_prob = fetch_player_over_probability(u["player"], _normalize_stat_name(u["stat"]), line, games=games)
        if true_over_prob is None:
            continue
        hold, imp_over, imp_under = remove_hold(over_price, under_price)
        ev_over = compute_ev(true_over_prob, over_price)
        ev_under = compute_ev(1 - true_over_prob, under_price)
        ev_props.append({
            "player": u["player"],
            "stat": u["stat"],
            "line": line,
            "over_odds": over_price,
            "under_odds": under_price,
            "book": ", ".join(books) if books else "consensus",
            "historical_over_prob": true_over_prob,
            "implied_over_prob": imp_over,
            "implied_under_prob": imp_under,
            "hold": hold,
            "ev_over": ev_over,
            "ev_under": ev_under,
        })
    return ev_props

# --------------- PARLAY SIMULATION -----------------
# Fixed payout multipliers for Underdog pick'em parlays
PICKEM_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0}


def _best_ev_side(prop):
    """Return the side of the prop with the higher EV and its probability."""
    if prop["ev_over"] >= prop["ev_under"]:
        return "over", prop["historical_over_prob"], prop["ev_over"]
    return "under", 1 - prop["historical_over_prob"], prop["ev_under"]


def generate_parlays(ev_props, min_legs=2, max_legs=5):
    """Generate parlays and calculate expected value for each."""
    from itertools import combinations

    edges = []
    for p in ev_props:
        side, prob, ev = _best_ev_side(p)
        if ev <= 0:
            continue
        edges.append({
            "player": p["player"],
            "stat": p["stat"],
            "side": side,
            "prob": prob,
        })

    parlays = []
    for n in range(min_legs, min(max_legs, len(edges)) + 1):
        payout = PICKEM_PAYOUTS.get(n)
        if not payout:
            continue
        for combo in combinations(edges, n):
            prob = 1.0
            legs_desc = []
            for leg in combo:
                prob *= leg["prob"]
                legs_desc.append(f"{leg['player']} {leg['stat']} {leg['side']}")
            ev = prob * payout - 1
            parlays.append({
                "legs": legs_desc,
                "num_legs": n,
                "prob": prob,
                "payout": payout,
                "ev": ev,
            })

    return sorted(parlays, key=lambda x: x["ev"], reverse=True)

# --------------- TELEGRAM ALERT -----------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except Exception as e:
        print("Telegram error:", e)
        return False

def alert_value_props(value_props):
    for prop in value_props:
        msg = (
            f"⚾️ MLB Value Prop!\n"
            f"{prop['player']} – {prop['stat']}\n"
            f"Underdog: {prop['underdog_line']} | Consensus: {prop['consensus_line']} ({'+' if prop['diff'] > 0 else ''}{prop['diff']:.2f})\n"
            f"Game: {prop['game']}\n"
            f"Book: {prop['book']}"
        )
        send_telegram_message(msg)

# --------------- TELEGRAM INTERACTIVITY -----------------
LAST_UPDATE_ID = None
latest_value_props = []


def get_telegram_updates(offset=None):
    """Fetch new Telegram updates starting from the given offset."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {}
    if offset is not None:
        params["offset"] = offset
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            return data.get("result", [])
    except Exception as e:
        print("Telegram update error:", e)
        return []


def process_telegram_commands():
    """Respond to user commands sent via Telegram."""
    global LAST_UPDATE_ID
    updates = get_telegram_updates(LAST_UPDATE_ID + 1 if LAST_UPDATE_ID else None)
    for upd in updates:
        LAST_UPDATE_ID = upd.get("update_id", LAST_UPDATE_ID)
        msg = upd.get("message", {})
        text = msg.get("text", "").strip()
        if not text:
            continue
        if text.startswith("/snapshot"):
            if latest_value_props:
                lines = [
                    f"{p['player']} {p['stat']} {p['underdog_line']} vs {p['consensus_line']} ({p['diff']:+.2f})"
                    for p in latest_value_props
                ]
                send_telegram_message("Current value props:\n" + "\n".join(lines))
            else:
                send_telegram_message("No value props found yet.")
        elif text.startswith("/history"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                send_telegram_message("Usage: /history LINE_ID")
                continue
            line_id = parts[1]
            rows = get_line_history(line_id)
            if rows:
                rows = rows[-5:]
                msgs = [
                    f"{datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')}: {line}"
                    for ts, line in rows
                ]
                send_telegram_message("\n".join(msgs))
            else:
                send_telegram_message(f"No history found for line id {line_id}")
        elif text.startswith("/help"):
            send_telegram_message(
                "Commands:\n/snapshot - current value props\n/history LINE_ID - recent line history"
            )
        else:
            send_telegram_message("Unknown command. Use /help for options.")

# --------------- MAIN LOOP -----------------
def main_loop(track_only: bool = False):
    """Continuous loop fetching props and optionally alerting Telegram."""
    if not track_only and not (
        THE_ODDS_API_KEY and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    ):
        raise RuntimeError(
            "API keys and Telegram credentials must be set via environment variables"
        )
    global latest_value_props
    init_db()
    while True:
        try:
            print("Fetching Underdog MLB props...")
            underdog_props = fetch_underdog_props()
            print(f"Fetched {len(underdog_props)} props.")
            save_line_history(underdog_props)
            print("Fetching consensus props...")
            consensus_props = fetch_consensus_props()
            print(f"Fetched consensus for {len(consensus_props)} games.")
            value_props = find_value_props(underdog_props, consensus_props)
            latest_value_props = value_props
            if value_props and not track_only:
                print(f"Found {len(value_props)} value props! Alerting Telegram.")
                alert_value_props(value_props)
            else:
                print("No value found this cycle.")
        except Exception as e:
            print("Main loop error:", e)
        process_telegram_commands()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Underdog MLB value finder")
    parser.add_argument(
        "--plot",
        metavar="LINE_ID",
        help="Plot line history for a given Underdog line id and exit",
    )
    parser.add_argument(
        "--track-only",
        action="store_true",
        help="Track line history without sending Telegram alerts",
    )
    parser.add_argument(
        "--ev-once",
        action="store_true",
        help="Calculate expected value using historical data and exit",
    )
    parser.add_argument(
        "--parlay-sim",
        action="store_true",
        help="Generate parlay EV simulation using historical probabilities and exit",
    )
    args = parser.parse_args()

    if args.plot:
        init_db()
        plot_line_history(args.plot)
    elif args.ev_once:
        print("Fetching Underdog MLB props...")
        ud_props = fetch_underdog_props()
        print("Fetching consensus props...")
        cons_props = fetch_consensus_props()
        ev_props = find_ev_props(ud_props, cons_props)
        for p in ev_props:
            print(
                f"{p['player']} {p['stat']} line {p['line']} "
                f"Over EV: {p['ev_over']:.3f} Under EV: {p['ev_under']:.3f} "
                f"Hold: {p['hold']:.3f}"
            )
    elif args.parlay_sim:
        print("Fetching Underdog MLB props...")
        ud_props = fetch_underdog_props()
        print("Fetching consensus props...")
        cons_props = fetch_consensus_props()
        ev_props = find_ev_props(ud_props, cons_props)
        parlays = generate_parlays(ev_props)
        for p in parlays[:10]:
            legs = " | ".join(p["legs"])
            print(
                f"{p['num_legs']}-leg parlay EV: {p['ev']:.3f} "
                f"TrueP: {p['prob']:.3f} Payout: {p['payout']}x -> {legs}"
            )
    else:
        main_loop(track_only=args.track_only)
