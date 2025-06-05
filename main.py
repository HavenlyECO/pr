import json
import urllib.request
import urllib.parse
import time
import os
import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
import random
import math

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

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
# Retry configuration for network requests
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds between retries
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

# Rotowire endpoints for lineup and injury data
ROTOWIRE_LINEUPS_URL = (
    "https://www.rotowire.com/dfs/services/mlb-lineups.php?format=json"
)
ROTOWIRE_INJURIES_URL = (
    "https://www.rotowire.com/dfs/services/mlb-injuries.php?format=json"
)

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

# Sets for managing lineup confirmations and scratches
LINEUP_PLAYERS = set()  # updated each cycle from Rotowire
MANUAL_CONFIRMED_PLAYERS = set()  # confirmed via Telegram
SCRATCHED_PLAYERS = set()  # scratched pitchers via Telegram

# Default bankroll configuration for unit sizing
BANKROLL = float(os.getenv("BANKROLL", "1000"))
# Base unit amount representing a typical wager size
BASE_UNIT = float(os.getenv("BASE_UNIT", "10"))

# ----- MACHINE LEARNING CONFIG -----
USE_ML_MODEL = os.getenv("USE_ML_MODEL", "0") == "1"
# Enable sequence models (LSTM/GRU) for stat value predictions
USE_SEQ_MODEL = os.getenv("USE_SEQ_MODEL", "0") == "1"

# Example logistic regression weights for advanced model
ML_WEIGHTS = [
    1.5,
    0.5,
    0.2,
    0.2,
    0.3,
    0.2,
    0.4,
    0.1,
    0.1,
    0.05,
]  # [stat_diff, park_emb1, park_emb2, weather, umpire, matchup, wind, temp, humidity]
ML_BIAS = 0.0

# Simplified environment factors by home team abbreviation
BALLPARK_FACTORS = {"COL": 1.2, "NYY": 1.1, "BOS": 1.05}
WEATHER_FACTORS = {"COL": 1.05, "NYY": 1.0, "BOS": 0.98}
UMPIRE_FACTORS = {"COL": 1.02, "NYY": 1.0, "BOS": 1.01}
WIND_FACTORS = {"COL": 1.05, "NYY": 0.95, "BOS": 1.0}
TEMP_FACTORS = {"COL": 1.02, "NYY": 1.0, "BOS": 0.99}
HUMIDITY_FACTORS = {"COL": 1.01, "NYY": 0.98, "BOS": 1.0}

# Simple 2D embedding for each ballpark
BALLPARK_EMBEDDINGS = {
    "COL": [0.5, 0.2],
    "NYY": [0.2, 0.1],
    "BOS": [0.1, 0.3],
}

# Opponent strength adjustments by team (1.0 = average)
TEAM_PITCHING_RATINGS = {"NYY": 1.05, "BOS": 0.95}
TEAM_HITTING_RATINGS = {"NYY": 1.1, "BOS": 1.0}

# Minimal mapping of player to team for matchup lookup
PLAYER_TEAM_MAP = {"Aaron Judge": "NYY", "Gerrit Cole": "NYY"}

# Example linear regression weights for predicting raw stat values
# Order of features: [recent_avg, park_emb1, park_emb2, ballpark, weather, umpire, matchup, wind, temp, humidity]
STAT_VALUE_WEIGHTS = {
    "hits": [0.85, 0.05, 0.05, 0.05, 0.03, 0.02, 0.05, 0.02, 0.02, 0.01],
    "strikeouts": [0.9, 0.04, 0.04, 0.03, 0.02, 0.0, 0.05, 0.02, 0.02, 0.01],
    "total bases": [0.8, 0.06, 0.04, 0.1, 0.03, 0.02, 0.05, 0.02, 0.02, 0.01],
}
# Bias terms for the stat prediction model
STAT_VALUE_BIAS = {"hits": 0.0, "strikeouts": 0.0, "total bases": 0.0}
DEFAULT_STAT_WEIGHTS = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01]




def init_db():
    """Initialize the SQLite database for line history and player data."""
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_performance (
            player TEXT,
            stat TEXT,
            game_date TEXT,
            value REAL,
            PRIMARY KEY (player, stat, game_date)
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


def save_player_performance(player: str, stat: str, logs):
    """Save a list of (date, value) tuples to the player performance table."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for game_date, value in logs:
        cur.execute(
            "INSERT OR REPLACE INTO player_performance (player, stat, game_date, value) VALUES (?, ?, ?, ?)",
            (player, stat, game_date, value),
        )
    conn.commit()
    conn.close()


def load_player_performance(player: str, stat: str, games: int):
    """Load recent game values for a player/stat from the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT game_date, value FROM player_performance WHERE player=? AND stat=? ORDER BY game_date DESC LIMIT ?",
        (player, stat, games),
    )
    rows = cur.fetchall()
    conn.close()
    return [r[1] for r in rows]


def fetch_player_game_logs_api(player_name: str, stat: str, games: int = 100, season: str = None):
    """Fetch game logs from the MLB Stats API."""
    try:
        search_url = (
            "https://statsapi.mlb.com/api/v1/people/search?name="
            + urllib.parse.quote(player_name)
        )
        with urllib.request.urlopen(search_url) as resp:
            search_data = json.loads(resp.read().decode())
        results = search_data.get("people") or []
        if not results:
            return []
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
    except Exception as exc:
        print("Historical data error:", exc)
        return []

    splits = data.get("stats", [{}])[0].get("splits", [])
    logs = []
    for g in splits:
        val = g.get("stat", {}).get(stat)
        if val is None:
            continue
        try:
            value = float(val)
        except Exception:
            continue
        game_date = g.get("date") or g.get("gameDate")
        logs.append((game_date, value))
    return logs


def get_recent_player_stats(player_name: str, stat: str, games: int = 100):
    """Return recent game values for the player, updating the DB if needed."""
    stats = load_player_performance(player_name, stat, games)
    if len(stats) >= games:
        return stats
    new_logs = fetch_player_game_logs_api(player_name, stat, games)
    if new_logs:
        save_player_performance(player_name, stat, new_logs)
        stats = load_player_performance(player_name, stat, games)
    return stats


def load_player_game_logs(player_name: str, stat: str, games: int = 100):
    """Return a list of (date, value) tuples for the player's recent games."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT game_date, value FROM player_performance WHERE player=? AND stat=? ORDER BY game_date DESC LIMIT ?",
        (player_name, stat, games),
    )
    rows = cur.fetchall()
    conn.close()
    if len(rows) >= games:
        return rows
    new_logs = fetch_player_game_logs_api(player_name, stat, games)
    if new_logs:
        save_player_performance(player_name, stat, new_logs)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT game_date, value FROM player_performance WHERE player=? AND stat=? ORDER BY game_date DESC LIMIT ?",
            (player_name, stat, games),
        )
        rows = cur.fetchall()
        conn.close()
    return rows


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


def get_environment_factors(game: str | None):
    """Return ballpark and weather-related factors for the given matchup."""
    if not game:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    try:
        parts = [p.strip() for p in game.split("@", 1)]
        home = parts[1] if len(parts) == 2 else parts[0]
    except Exception:
        home = game.strip()
    return (
        BALLPARK_FACTORS.get(home, 1.0),
        WEATHER_FACTORS.get(home, 1.0),
        UMPIRE_FACTORS.get(home, 1.0),
        WIND_FACTORS.get(home, 1.0),
        TEMP_FACTORS.get(home, 1.0),
        HUMIDITY_FACTORS.get(home, 1.0),
    )


def get_ballpark_embedding(game: str | None) -> list[float]:
    """Return embedding vector representing the ballpark."""
    if not game:
        return [0.0, 0.0]
    try:
        parts = [p.strip() for p in game.split("@", 1)]
        home = parts[1] if len(parts) == 2 else parts[0]
    except Exception:
        home = game.strip()
    return BALLPARK_EMBEDDINGS.get(home, [0.0, 0.0])


def get_matchup_factor(player_name: str, stat: str, game: str | None) -> float:
    """Return matchup factor based on opposing team strength."""
    if not game:
        return 1.0
    try:
        away, home = [p.strip() for p in game.split("@", 1)]
    except Exception:
        return 1.0
    team = PLAYER_TEAM_MAP.get(player_name)
    if not team:
        return 1.0
    if team == away:
        opp = home
    elif team == home:
        opp = away
    else:
        return 1.0

    norm = _normalize_stat_name(stat)
    pitching_stats = {"strikeouts", "outs", "earned runs", "hits allowed"}
    if norm in pitching_stats or "pitch" in norm:
        return TEAM_HITTING_RATINGS.get(opp, 1.0)
    return TEAM_PITCHING_RATINGS.get(opp, 1.0)


def predict_over_probability_ml(
    player_name: str, stat: str, line: float, game: str | None = None
):
    """Predict over probability using a simple logistic regression model."""
    stats = get_recent_player_stats(player_name, stat, 30)
    if not stats:
        return None
    avg_stat = sum(stats) / len(stats)
    ballpark, weather, umpire, wind, temp, humidity = get_environment_factors(game)
    park_emb = get_ballpark_embedding(game)
    matchup = get_matchup_factor(player_name, stat, game)
    features = [
        avg_stat - line,
        *park_emb,
        weather - 1.0,
        umpire - 1.0,
        matchup - 1.0,
        wind - 1.0,
        temp - 1.0,
        humidity - 1.0,
    ]
    z = ML_BIAS + sum(w * f for w, f in zip(ML_WEIGHTS, features))
    return 1 / (1 + math.exp(-z))


def _predict_stat_sequence(values: list[float], epochs: int = 50):
    """Train a simple LSTM on the sequence and forecast the next value."""
    if torch is None or nn is None:
        return None
    if len(values) < 6:
        return None
    seq_len = min(5, len(values) - 1)
    x_data = []
    y_data = []
    for i in range(len(values) - seq_len):
        x_data.append(values[i : i + seq_len])
        y_data.append(values[i + seq_len])
    if not x_data:
        return None
    X = torch.tensor(x_data, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y_data, dtype=torch.float32)

    class LSTMReg(nn.Module):
        def __init__(self, hidden_size: int = 16):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out).squeeze(-1)

    model = LSTMReg()
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()

    model.eval()
    inp = torch.tensor(values[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(inp).item()
    return pred


def predict_stat_value_ml(player_name: str, stat: str, game: str | None = None):
    """Predict the raw stat value using ML models."""
    stats = get_recent_player_stats(player_name, stat, 30)
    if not stats:
        return None

    if USE_SEQ_MODEL:
        try:
            pred = _predict_stat_sequence(stats)
            if pred is not None:
                return pred
        except Exception as exc:
            print("Sequence model error:", exc)

    avg_stat = sum(stats) / len(stats)
    ballpark, weather, umpire, wind, temp, humidity = get_environment_factors(game)
    park_emb = get_ballpark_embedding(game)
    matchup = get_matchup_factor(player_name, stat, game)
    key = _normalize_stat_name(stat)
    weights = STAT_VALUE_WEIGHTS.get(key, DEFAULT_STAT_WEIGHTS)
    bias = STAT_VALUE_BIAS.get(key, 0.0)
    features = [avg_stat, *park_emb, ballpark, weather, umpire, matchup, wind, temp, humidity]
    return bias + sum(w * f for w, f in zip(weights, features))


def fetch_player_over_probability(
    player_name: str,
    stat: str,
    line: float,
    season: str = None,
    games: int = 100,
    game: str | None = None,
):
    """Return probability of player going over a line.

    Uses a machine learning model when enabled, otherwise falls back to simple
    historical frequency.
    """

    if USE_ML_MODEL:
        prob = predict_over_probability_ml(player_name, stat, line, game)
        if prob is not None:
            return prob

    stats = get_recent_player_stats(player_name, stat, games)
    if not stats:
        return None
    over = sum(1 for v in stats if v > line)
    return over / len(stats)

# --------------- FETCH UNDERDOG MLB PROPS -----------------
def _notify_error(message: str):
    """Print and send an error notification via Telegram if configured."""
    print(message)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram_message(f"[ERROR] {message}")
        except Exception as exc:
            print("Telegram notify error:", exc)


def _parse_underdog_data(data):
    """Convert raw Underdog JSON to a list of prop dictionaries."""
    props = []
    for line in data.get("over_under_lines", []):
        prop = {
            "player": line.get("over_under", {}).get("title"),
            "stat": line.get("over_under", {}).get("stat_type"),
            "line": line.get("line_score"),
            "game": line.get("over_under", {}).get("game", {}).get("matchup"),
            "id": line.get("id"),
        }
        if prop["player"] and prop["stat"] and prop["line"]:
            props.append(prop)
    return props


def _fetch_json_with_retry(url: str, headers: dict | None = None) -> dict:
    """Fetch JSON from a URL with retry logic."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP error: {resp.status}")
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_exc = exc
            time.sleep(RETRY_BACKOFF * attempt)
    raise last_exc


def scrape_underdog_props() -> list:
    """Fallback scraper for Underdog props when the API fails."""
    params = {"sport": "mlb"}
    query_params = urllib.parse.urlencode(params)
    url = f"{UNDERDOG_GQL_URL}?{query_params}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        data = _fetch_json_with_retry(url, headers=headers)
    except Exception as exc:
        _notify_error(f"Failover scrape error: {exc}")
        return []
    return _parse_underdog_data(data)


def fetch_underdog_props() -> list:
    params = {"sport": "mlb", "platform": "web"}
    query_params = urllib.parse.urlencode(params)
    url = f"{UNDERDOG_GQL_URL}?{query_params}"
    try:
        data = _fetch_json_with_retry(url)
        return _parse_underdog_data(data)
    except Exception as exc:
        _notify_error(f"Underdog fetch failed: {exc}")
        return scrape_underdog_props()

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

# --------------- ROTOWIRE DATA -----------------
def fetch_rotowire_lineup_players():
    """Return a set of player names expected to start today."""
    try:
        with urllib.request.urlopen(ROTOWIRE_LINEUPS_URL) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print("Rotowire lineup fetch error:", exc)
        return set()

    players = set()
    for game in data.get("games", []):
        for side in ("away", "home"):
            lineup = game.get(side, {}).get("lineup", [])
            for p in lineup:
                name = p.get("player") or p.get("name")
                if name:
                    players.add(name.strip())
    return players


def fetch_rotowire_injured_players():
    """Return a set of players listed as out."""
    try:
        with urllib.request.urlopen(ROTOWIRE_INJURIES_URL) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print("Rotowire injury fetch error:", exc)
        return set()

    players = set()
    for team in data.get("teams", []):
        for p in team.get("players", []):
            status = p.get("status", "").lower()
            if "out" in status or "dl" in status or "il" in status:
                name = p.get("name") or p.get("player")
                if name:
                    players.add(name.strip())
    return players


def filter_dead_props(props, lineup_players, injured_players):
    """Remove props for players not starting or injured."""
    if not lineup_players and not injured_players:
        return props
    norm_lineup = {_normalize_name(p) for p in lineup_players}
    norm_injured = {_normalize_name(p) for p in injured_players}
    filtered = []
    for p in props:
        norm = _normalize_name(p.get("player", ""))
        if norm in norm_injured:
            continue
        if norm_lineup and norm not in norm_lineup:
            continue
        filtered.append(p)
    return filtered

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
        true_over_prob = fetch_player_over_probability(
            u["player"],
            _normalize_stat_name(u["stat"]),
            line,
            games=games,
            game=u.get("game"),
        )
        if true_over_prob is None:
            continue
        hold, imp_over, imp_under = remove_hold(over_price, under_price)
        ev_over = compute_ev(true_over_prob, over_price)
        ev_under = compute_ev(1 - true_over_prob, under_price)
        ev_props.append({
            "player": u["player"],
            "stat": u["stat"],
            "line": line,
            "underdog_line": u["line"],
            "consensus_line": line,
            "diff": float(u["line"]) - float(line),
            "over_odds": over_price,
            "under_odds": under_price,
            "book": ", ".join(books) if books else "consensus",
            "game": u.get("game"),
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


def kelly_fraction(win_prob: float, odds: float) -> float:
    """Return optimal bankroll fraction for a bet using the Kelly criterion."""
    b = american_to_decimal(odds) - 1
    q = 1 - win_prob
    if b <= 0:
        return 0.0
    fraction = (b * win_prob - q) / b
    return max(fraction, 0.0)


def simulate_drawdown(ev_props, bankroll=1000.0, fraction=1.0, sims=1000):
    """Simulate betting using Kelly sizing and return avg bankroll and drawdown."""
    bets = []
    for p in ev_props:
        side, prob, _ = _best_ev_side(p)
        odds = p["over_odds"] if side == "over" else p["under_odds"]
        kelly = kelly_fraction(prob, odds) * fraction
        if kelly <= 0:
            continue
        bets.append({"prob": prob, "odds": odds, "kelly": kelly})

    if not bets:
        return bankroll, 0.0

    final_bankrolls = []
    drawdowns = []
    for _ in range(sims):
        bal = bankroll
        peak = bankroll
        max_dd = 0.0
        for b in bets:
            wager = bal * b["kelly"]
            if random.random() < b["prob"]:
                bal += wager * (american_to_decimal(b["odds"]) - 1)
            else:
                bal -= wager
            peak = max(peak, bal)
            dd = 1 - bal / peak
            max_dd = max(max_dd, dd)
        final_bankrolls.append(bal)
        drawdowns.append(max_dd)

    avg_final = sum(final_bankrolls) / sims
    avg_dd = sum(drawdowns) / sims
    return avg_final, avg_dd


def calculate_unit_size(prop, bankroll=BANKROLL, base_unit=BASE_UNIT):
    """Return recommended stake for a prop using edge and volatility."""
    side, prob, _ = _best_ev_side(prop)
    odds = prop["over_odds"] if side == "over" else prop["under_odds"]
    kelly = kelly_fraction(prob, odds)
    stat_norm = _normalize_stat_name(prop["stat"])
    stat_threshold = EDGE_THRESHOLD_BY_STAT.get(stat_norm, DEFAULT_EDGE_THRESHOLD)
    edge = abs(prop.get("diff", 0))
    edge_factor = min(edge / stat_threshold, 2.0)
    volatility_factor = DEFAULT_EDGE_THRESHOLD / stat_threshold
    fraction = kelly * edge_factor * volatility_factor
    stake = bankroll * fraction / base_unit
    return side, max(stake, 0.0)


def _get_stat_on_date(player: str, stat: str, game_date: str):
    """Return stat value for a player on a specific date.

    If the value is not already cached in the DB it will attempt to fetch
    logs from the MLB Stats API and store them for future use.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT value FROM player_performance WHERE player=? AND stat=? AND game_date=?",
        (player, stat, game_date),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]

    logs = fetch_player_game_logs_api(player, stat, games=200, season=game_date[:4])
    if logs:
        save_player_performance(player, stat, logs)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT value FROM player_performance WHERE player=? AND stat=? AND game_date=?",
            (player, stat, game_date),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
    return None


def _pearson_corr(a, b):
    """Return Pearson correlation coefficient for two sequences."""
    n = len(a)
    if n < 2:
        return 0.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def same_game_parlay_correlation(leg_a, leg_b, games: int = 50):
    """Return correlation between two prop outcomes from the same game."""
    stat_a = _normalize_stat_name(leg_a["stat"])
    stat_b = _normalize_stat_name(leg_b["stat"])
    logs_a = load_player_game_logs(leg_a["player"], stat_a, games * 2)
    logs_b = load_player_game_logs(leg_b["player"], stat_b, games * 2)
    dict_a = {d: v for d, v in logs_a}
    dict_b = {d: v for d, v in logs_b}
    common = [d for d in dict_a if d in dict_b]
    if len(common) < 10:
        return 0.0
    hits_a = []
    hits_b = []
    for d in common[:games]:
        va = dict_a[d]
        vb = dict_b[d]
        ha = va > leg_a["line"] if leg_a["side"] == "over" else va < leg_a["line"]
        hb = vb > leg_b["line"] if leg_b["side"] == "over" else vb < leg_b["line"]
        hits_a.append(1 if ha else 0)
        hits_b.append(1 if hb else 0)
    return _pearson_corr(hits_a, hits_b)


CORRELATION_WEIGHT = 0.25


def apply_same_game_correlation(legs, base_prob):
    """Adjust parlay probability using same-game correlations."""
    prob = base_prob
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            g1 = legs[i].get("game")
            if g1 and g1 == legs[j].get("game"):
                corr = same_game_parlay_correlation(legs[i], legs[j])
                prob *= 1 + corr * CORRELATION_WEIGHT
    return max(0.0, min(prob, 1.0))


def backtest_line_history():
    """Run a simple backtest across stored line history."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT line_id, player, stat, line, ts FROM line_history")
    rows = cur.fetchall()
    conn.close()

    results = []
    for line_id, player, stat, line, ts in rows:
        game_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        stat_norm = _normalize_stat_name(stat)
        value = _get_stat_on_date(player, stat_norm, game_date)
        if value is None:
            continue
        outcome = "over" if value > line else "under"
        diff = value - line
        results.append((outcome, diff))

    if not results:
        print("No historical data available for backtest.")
        return

    over_hits = sum(1 for o, _ in results if o == "over")
    under_hits = len(results) - over_hits
    avg_diff = sum(d for _, d in results) / len(results)

    print(f"Backtested {len(results)} props")
    print(f"Over win rate: {over_hits/len(results):.3f}")
    print(f"Under win rate: {under_hits/len(results):.3f}")
    print(f"Avg result minus line: {avg_diff:.2f}")


def generate_parlays(ev_props, min_legs=2, max_legs=5, use_corr=True):
    """Generate parlays and calculate expected value for each.

    When ``use_corr`` is True, same-game correlations between legs are
    measured using historical game logs and applied to adjust the joint
    probability of the parlay.
    """
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
            "game": p.get("game"),
            "line": p["line"],
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
            if use_corr:
                prob = apply_same_game_correlation(combo, prob)
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
        norm = _normalize_name(prop.get("player", ""))
        confirmed = norm in MANUAL_CONFIRMED_PLAYERS or norm in LINEUP_PLAYERS
        if not confirmed or norm in SCRATCHED_PLAYERS:
            continue
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
    global LAST_UPDATE_ID, MANUAL_CONFIRMED_PLAYERS, SCRATCHED_PLAYERS
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
                "Commands:\n"
                "/snapshot - current value props\n"
                "/history LINE_ID - recent line history\n"
                "/confirm NAME - manually confirm a player\n"
                "/scratch NAME - mark a player scratched\n"
                "/units - show suggested bet sizing"
            )
        elif text.startswith("/confirm"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                send_telegram_message("Usage: /confirm PLAYER_NAME")
                continue
            name = _normalize_name(parts[1])
            MANUAL_CONFIRMED_PLAYERS.add(name)
            if name in SCRATCHED_PLAYERS:
                SCRATCHED_PLAYERS.discard(name)
            send_telegram_message(f"Confirmed {parts[1]} for alerts.")
        elif text.startswith("/scratch"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                send_telegram_message("Usage: /scratch PLAYER_NAME")
                continue
            name = _normalize_name(parts[1])
            SCRATCHED_PLAYERS.add(name)
            if name in MANUAL_CONFIRMED_PLAYERS:
                MANUAL_CONFIRMED_PLAYERS.discard(name)
            send_telegram_message(f"Marked {parts[1]} as scratched.")
        elif text.startswith("/units"):
            ud_props = fetch_underdog_props()
            cons_props = fetch_consensus_props()
            ev_props = find_ev_props(ud_props, cons_props)
            msgs = []
            for p in ev_props[:5]:
                side, units = calculate_unit_size(p)
                if units <= 0:
                    continue
                msgs.append(
                    f"{p['player']} {p['stat']} {side} - {units:.2f} units"
                )
            if msgs:
                send_telegram_message("\n".join(msgs))
            else:
                send_telegram_message("No bets meet sizing criteria.")
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
            print("Fetching Rotowire injuries and lineups...")
            lineup_players = fetch_rotowire_lineup_players()
            injured_players = fetch_rotowire_injured_players()
            global LINEUP_PLAYERS
            LINEUP_PLAYERS = {_normalize_name(p) for p in lineup_players}
            before = len(underdog_props)
            underdog_props = filter_dead_props(
                underdog_props, lineup_players, injured_players
            )
            if len(underdog_props) != before:
                print(
                    f"Filtered to {len(underdog_props)} props after injury/lineup check."
                )
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
    parser.add_argument(
        "--risk-sim",
        action="store_true",
        help="Simulate bankroll growth with Kelly staking and report drawdown",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run historical backtest using stored line data",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll for risk simulation",
    )
    parser.add_argument(
        "--kelly-scale",
        type=float,
        default=1.0,
        help="Scale Kelly fraction (e.g. 0.5 for half Kelly)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations for risk analysis",
    )
    parser.add_argument(
        "--predict-stat",
        nargs=3,
        metavar=("PLAYER", "STAT", "GAME"),
        help="Predict a player's stat value for a matchup and exit",
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
    elif args.risk_sim:
        print("Fetching Underdog MLB props...")
        ud_props = fetch_underdog_props()
        print("Fetching consensus props...")
        cons_props = fetch_consensus_props()
        ev_props = find_ev_props(ud_props, cons_props)
        avg_final, avg_dd = simulate_drawdown(
            ev_props,
            bankroll=args.bankroll,
            fraction=args.kelly_scale,
            sims=args.sims,
        )
        print(
            f"Avg Final Bankroll: {avg_final:.2f} | Avg Max Drawdown: {avg_dd*100:.1f}%"
        )
    elif args.predict_stat:
        player, stat, game = args.predict_stat
        init_db()
        value = predict_stat_value_ml(player, stat, game)
        if value is None:
            print("Insufficient data to make prediction")
        else:
            print(f"Predicted {stat} for {player} vs {game}: {value:.2f}")
    elif args.backtest:
        backtest_line_history()
    else:
        main_loop(track_only=args.track_only)
