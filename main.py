import json
import urllib.request
import urllib.parse
import time
import os

# --------------- CONFIG -----------------
UNDERDOG_GQL_URL = "https://api.underdogfantasy.com/beta/v4/over_under_lines"
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CHECK_INTERVAL = 60  # seconds between checks
EDGE_THRESHOLD = 0.7 # minimum difference to alert (tune for MLB stat, e.g. 0.5-1)

if not (THE_ODDS_API_KEY and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
    raise RuntimeError(
        "API keys and Telegram credentials must be set via environment variables"
    )

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
        "markets": "player_hits,player_home_runs,player_total_bases,player_rbis,player_strikeouts,player_runs", # add more as needed
        "oddsFormat": "american"
    }
    query_params = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query_params}") as resp:
        if resp.status != 200:
            raise RuntimeError(f"TheOdds API error: {resp.status}")
        return json.loads(resp.read().decode())

# --------------- FIND VALUE PROPS -----------------
def find_value_props(ud_props, cons_props, threshold=EDGE_THRESHOLD):
    value_props = []
    for u in ud_props:
        u_player = u['player'].split()[0]  # First name only for rough match (you can improve this)
        u_stat = u['stat'].lower()
        for game in cons_props:
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    # Fuzzy stat matching
                    if u_stat in market_key:
                        for outcome in market.get("outcomes", []):
                            outcome_name = outcome.get("name", "")
                            # Simple player name check (first name substring)
                            if u_player.lower() in outcome_name.lower():
                                consensus_line = outcome.get("point")
                                if consensus_line is not None:
                                    try:
                                        diff = float(u["line"]) - float(consensus_line)
                                        if abs(diff) >= threshold:
                                            value_props.append({
                                                "player": u["player"],
                                                "stat": u["stat"],
                                                "underdog_line": u["line"],
                                                "consensus_line": consensus_line,
                                                "diff": diff,
                                                "game": u["game"],
                                                "book": bookmaker.get("title", "consensus")
                                            })
                                    except Exception as e:
                                        print("Parse error:", e)
    return value_props

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

# --------------- MAIN LOOP -----------------
def main_loop():
    while True:
        try:
            print("Fetching Underdog MLB props...")
            underdog_props = fetch_underdog_props()
            print(f"Fetched {len(underdog_props)} props.")
            print("Fetching consensus props...")
            consensus_props = fetch_consensus_props()
            print(f"Fetched consensus for {len(consensus_props)} games.")
            value_props = find_value_props(underdog_props, consensus_props)
            if value_props:
                print(f"Found {len(value_props)} value props! Alerting Telegram.")
                alert_value_props(value_props)
            else:
                print("No value found this cycle.")
        except Exception as e:
            print("Main loop error:", e)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()
