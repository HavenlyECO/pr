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
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------- CONFIG -----------------
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
DEFAULT_BOOKMAKER = "draftkings"
DB_PATH = os.getenv("LINE_HISTORY_DB", "line_history.db")
CHECK_INTERVAL = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 5
DEFAULT_EDGE_THRESHOLD = 0.7
EDGE_THRESHOLD_BY_STAT = {
    "strikeouts": 0.5,
    "total bases": 1.5,
}
EDGE_THRESHOLD = DEFAULT_EDGE_THRESHOLD
PLAYER_MATCH_THRESHOLD = 70
BOOKMAKER_KEYS = ["draftkings", "fanduel", "pointsbetus", "betmgm"]

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
REVERSE_STAT_KEY_MAP = {m: s for s, ms in STAT_KEY_MAP.items() for m in ms}
ALL_MARKETS = ",".join(sorted({m for v in STAT_KEY_MAP.values() for m in v}))

BANKROLL = float(os.getenv("BANKROLL", "1000"))
BASE_UNIT = float(os.getenv("BASE_UNIT", "10"))

USE_ML_MODEL = os.getenv("USE_ML_MODEL", "0") == "1"
USE_SEQ_MODEL = os.getenv("USE_SEQ_MODEL", "0") == "1"

ML_WEIGHTS = [
    1.5, 0.5, 0.2, 0.2, 0.3, 0.2, 0.4, 0.1, 0.1, 0.05,
]
ML_BIAS = 0.0
BALLPARK_FACTORS = {"COL": 1.2, "NYY": 1.1, "BOS": 1.05}
WEATHER_FACTORS = {"COL": 1.05, "NYY": 1.0, "BOS": 0.98}
UMPIRE_FACTORS = {"COL": 1.02, "NYY": 1.0, "BOS": 1.01}
WIND_FACTORS = {"COL": 1.05, "NYY": 0.95, "BOS": 1.0}
TEMP_FACTORS = {"COL": 1.02, "NYY": 1.0, "BOS": 0.99}
HUMIDITY_FACTORS = {"COL": 1.01, "NYY": 0.98, "BOS": 1.0}
BALLPARK_EMBEDDINGS = {
    "COL": [0.5, 0.2],
    "NYY": [0.2, 0.1],
    "BOS": [0.1, 0.3],
}
TEAM_PITCHING_RATINGS = {"NYY": 1.05, "BOS": 0.95}
TEAM_HITTING_RATINGS = {"NYY": 1.1, "BOS": 1.0}
PLAYER_TEAM_MAP = {"Aaron Judge": "NYY", "Gerrit Cole": "NYY"}
STAT_VALUE_WEIGHTS = {
    "hits": [0.85, 0.05, 0.05, 0.05, 0.03, 0.02, 0.05, 0.02, 0.02, 0.01],
    "strikeouts": [0.9, 0.04, 0.04, 0.03, 0.02, 0.0, 0.05, 0.02, 0.02, 0.01],
    "total bases": [0.8, 0.06, 0.04, 0.1, 0.03, 0.02, 0.05, 0.02, 0.02, 0.01],
}
STAT_VALUE_BIAS = {"hits": 0.0, "strikeouts": 0.0, "total bases": 0.0}
DEFAULT_STAT_WEIGHTS = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01]

# --------------- DB FUNCTIONS -----------------
def init_db():
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
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, line FROM line_history WHERE line_id=? ORDER BY ts",
        (line_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def save_player_performance(player, stat, logs):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for game_date, value in logs:
        cur.execute(
            "INSERT OR REPLACE INTO player_performance (player, stat, game_date, value) VALUES (?, ?, ?, ?)",
            (player, stat, game_date, value),
        )
    conn.commit()
    conn.close()

def load_player_performance(player, stat, games):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT game_date, value FROM player_performance WHERE player=? AND stat=? ORDER BY game_date DESC LIMIT ?",
        (player, stat, games),
    )
    rows = cur.fetchall()
    conn.close()
    return [r[1] for r in rows]

# --------------- ODDS API FUNCTIONS -----------------
def _fetch_json_with_retry(url, headers=None):
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

def fetch_sports():
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={THE_ODDS_API_KEY}"
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"TheOdds API error: {resp.status}")
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"Sports fetch failed: {exc}")
        return []

def get_valid_mlb_markets():
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"TheOdds API error: {resp.status}")
            data = json.loads(resp.read().decode())
            return data.get("markets", [])
    except Exception as exc:
        print("Could not fetch valid markets for MLB:", exc)
        return ["h2h"]

def fetch_consensus_props():
    sports = fetch_sports()
    mlb = next((s for s in sports if s.get("key") == "baseball_mlb"), None)
    if not mlb:
        print("baseball_mlb sport not found")
        return []
    valid_markets = get_valid_mlb_markets()
    requested_markets = [m for m in ALL_MARKETS.split(",") if m in valid_markets]
    if not requested_markets:
        requested_markets = ["h2h"]
    url = f"https://api.the-odds-api.com/v4/sports/{mlb['key']}/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(requested_markets),
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKER_KEYS),
    }
    query_params = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_params}"
    print("CONSENSUS URL:", full_url)
    try:
        with urllib.request.urlopen(full_url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"TheOdds API error: {resp.status}")
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"Consensus fetch failed: {exc}")
        return []

def fetch_baseline_props():
    cons_data = fetch_consensus_props()
    props = []
    for game in cons_data:
        matchup = None
        away = game.get("away_team")
        home = game.get("home_team")
        if away and home:
            matchup = f"{away} @ {home}"
        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") != DEFAULT_BOOKMAKER:
                continue
            for market in bookmaker.get("markets", []):
                stat = REVERSE_STAT_KEY_MAP.get(market.get("key"))
                if not stat:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    if "over" not in name.lower():
                        continue
                    player = re.sub(r"\s+over\s*$", "", name, flags=re.I)
                    line = outcome.get("point")
                    if player and line is not None:
                        props.append({
                            "player": player,
                            "stat": stat,
                            "line": line,
                            "game": matchup,
                            "id": f"{DEFAULT_BOOKMAKER}-{player}-{stat}",
                        })
                    break
    return props

# --------------- STATS & MATCHING -----------------
def _normalize_name(name):
    name = name.lower().replace('.', '').replace(',', '')
    for suffix in [" jr", " sr", " iii", " ii"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()

def _normalize_stat_name(stat):
    return re.sub(r"[^a-z0-9]+", " ", stat.lower()).strip()

def get_market_keys(stat):
    norm = _normalize_stat_name(stat)
    if norm in STAT_KEY_MAP:
        return STAT_KEY_MAP[norm]
    for key, markets in STAT_KEY_MAP.items():
        if key in norm:
            return markets
    return []

def player_match(name_a, name_b, threshold=PLAYER_MATCH_THRESHOLD):
    score = SequenceMatcher(None, _normalize_name(name_a), _normalize_name(name_b)).ratio() * 100
    return score >= threshold

# --------------- ML/STAT PREDICTION -----------------
def get_environment_factors(game):
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

def get_ballpark_embedding(game):
    if not game:
        return [0.0, 0.0]
    try:
        parts = [p.strip() for p in game.split("@", 1)]
        home = parts[1] if len(parts) == 2 else parts[0]
    except Exception:
        home = game.strip()
    return BALLPARK_EMBEDDINGS.get(home, [0.0, 0.0])

def get_matchup_factor(player_name, stat, game):
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

def predict_over_probability_ml(player_name, stat, line, game=None):
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

def fetch_player_game_logs_api(player_name, stat, games=100, season=None):
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

def get_recent_player_stats(player_name, stat, games=100):
    stats = load_player_performance(player_name, stat, games)
    if len(stats) >= games:
        return stats
    new_logs = fetch_player_game_logs_api(player_name, stat, games)
    if new_logs:
        save_player_performance(player_name, stat, new_logs)
        stats = load_player_performance(player_name, stat, games)
    return stats

def fetch_player_over_probability(player_name, stat, line, season=None, games=100, game=None):
    if USE_ML_MODEL:
        prob = predict_over_probability_ml(player_name, stat, line, game)
        if prob is not None:
            return prob
    stats = get_recent_player_stats(player_name, stat, games)
    if not stats:
        return None
    over = sum(1 for v in stats if v > line)
    return over / len(stats)

# --------------- FIND VALUE PROPS -----------------
def aggregate_consensus_line(cons_props, market_keys, player_name):
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

def find_value_props(ud_props, cons_props, threshold=EDGE_THRESHOLD):
    value_props = []
    for u in ud_props:
        market_keys = get_market_keys(u["stat"])
        if not market_keys:
            continue
        consensus_line, books = aggregate_consensus_line(cons_props, market_keys, u["player"])
        if consensus_line is not None:
            try:
                diff = float(u["line"]) - float(consensus_line)
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

# --------------- MAIN LOOP -----------------
def main_loop():
    init_db()
    while True:
        try:
            print("Fetching MLB props...")
            underdog_props = fetch_baseline_props()
            print(f"Fetched {len(underdog_props)} props.")
            save_line_history(underdog_props)
            print("Fetching consensus props...")
            consensus_props = fetch_consensus_props()
            print(f"Fetched consensus for {len(consensus_props)} games.")
            value_props = find_value_props(underdog_props, consensus_props)
            if value_props:
                print(f"Found {len(value_props)} value props!")
                for prop in value_props:
                    print(
                        f"{prop['player']} – {prop['stat']}: "
                        f"Underdog {prop['underdog_line']} vs Consensus {prop['consensus_line']} "
                        f"Diff: {prop['diff']:+.2f} Game: {prop['game']} Book: {prop['book']}"
                    )
            else:
                print("No value found this cycle.")
        except Exception as e:
            print("Main loop error:", e)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    print("Starting MLB Odds Value Finder...")
    main_loop()
