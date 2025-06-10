import os
import json
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime

API_KEY = os.getenv("THE_ODDS_API_KEY")
SCORES_HISTORY_FILE = Path("h2h_data") / "scores_history.jsonl"


def build_scores_url(sport_key: str, days_from: int = 0, date_format: str = "iso") -> str:
    """Return API URL for the scores endpoint."""
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    params = {"apiKey": API_KEY}
    if days_from:
        params["daysFrom"] = str(days_from)
    if date_format:
        params["dateFormat"] = date_format
    return f"{base}?{urllib.parse.urlencode(params)}"


def fetch_scores(sport_key: str, days_from: int = 0, date_format: str = "iso") -> list:
    """Fetch upcoming, live and recent scores for ``sport_key``."""
    if not API_KEY:
        print("THE_ODDS_API_KEY environment variable is not set; cannot fetch scores")
        return []
    url = build_scores_url(sport_key, days_from=days_from, date_format=date_format)
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # pragma: no cover - network error handling
        print(f"Error fetching scores: {exc}")
        return []


def append_scores_history(scores: list, path: Path = SCORES_HISTORY_FILE) -> None:
    """Append ``scores`` to ``path`` as JSON lines with a timestamp."""
    if not scores:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    with open(path, "a") as f:
        for game in scores:
            record = {"timestamp": timestamp, **game}
            json.dump(record, f)
            f.write("\n")

