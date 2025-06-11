import os
import json
import pickle
import time
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import urllib.error
import hashlib
from contextlib import contextmanager, nullcontext

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None

try:  # Optional social media clients
    import praw
except ImportError:  # pragma: no cover - optional dependency
    praw = None

try:
    import tweepy
except ImportError:  # pragma: no cover - optional dependency
    tweepy = None

try:
    from telethon import TelegramClient
except ImportError:  # pragma: no cover - optional dependency
    TelegramClient = None

try:  # Optional dependency for memory profiling
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required. Install it with 'pip install python-dotenv'")

# Define this class here so it can be unpickled
class SimpleOddsModel:
    """A model that converts American odds to implied probability."""

    def predict_proba(self, X):
        price1 = X["price1"].values[0]
        if price1 > 0:
            prob = 100 / (price1 + 100)
        else:
            prob = abs(price1) / (abs(price1) + 100)
        return np.array([[1 - prob, prob]])

def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to an implied win probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def american_odds_to_payout(odds: float) -> float:
    """Return the profit on a $1 bet for the given American odds."""
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


LINE_MOVEMENT_THRESHOLD = 15


@contextmanager
def memory_usage(section: str):
    """Log memory usage delta for a code block when ``psutil`` is installed."""
    if psutil is None:
        yield
        return
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 ** 2)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_mem = process.memory_info().rss / (1024 ** 2)
        duration = time.perf_counter() - start_time
        print(
            f"[mem:{section}] {end_mem - start_mem:+.1f} MB in {duration:.1f}s (now {end_mem:.1f} MB)"
        )

def line_movement_delta(opening_odds: float, current_odds: float) -> float:
    """Return ``opening_odds - current_odds``.

    Positive values indicate the line has shortened (market confidence
    increasing) while negative values show the odds have lengthened.
    """
    return opening_odds - current_odds


def line_movement_flag(delta: float, threshold: float = LINE_MOVEMENT_THRESHOLD) -> int:
    """Return 1, -1 or 0 depending on whether ``delta`` exceeds ``threshold``.

    ``1`` means the line moved in the team's favour by at least ``threshold``
    points, ``-1`` means it drifted against the team by that amount and ``0``
    indicates no significant movement.
    """
    if delta >= threshold:
        return 1
    if delta <= -threshold:
        return -1
    return 0

def compute_recency_weights(dates: pd.Series, half_life_days: float = 30.0) -> pd.Series:
    """Return exponentially decayed weights based on recency.

    The most recent date receives weight 1.0 and each ``half_life_days`` offset
    halves the weight. Missing dates result in uniform weight 1.0.
    """
    if dates.isnull().all():
        return pd.Series(1.0, index=dates.index)
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.isnull().all():
        return pd.Series(1.0, index=dates.index)
    latest = parsed.max()
    age = (latest - parsed).dt.days.fillna(0)
    weights = 0.5 ** (age / float(half_life_days))
    return weights


def fetch_team_stats(team: str) -> dict:
    """Return statistics dictionary for ``team``.

    This placeholder implementation returns an empty mapping. Replace it with
    real data retrieval as needed.
    """

    return {}


def enrich_game_data(row: dict) -> dict:
    """Add additional features needed by advanced ML models."""

    team1 = row.get("team1")
    team2 = row.get("team2")

    team1_stats = fetch_team_stats(team1)
    team2_stats = fetch_team_stats(team2)

    row.update(
        {
            "pregame_price": row.get("price1"),
            "pregame_line": row.get("price1"),
            "home_team": team1,
            "away_team": team2,
            "day_night": "D",
            "game_day": datetime.utcnow().weekday(),
            "is_weekend": datetime.utcnow().weekday() >= 5,
        }
    )

    row.update({f"team1_{k}": v for k, v in team1_stats.items()})
    row.update({f"team2_{k}": v for k, v in team2_stats.items()})

    return row


def public_fade_flag(ticket_percent: float, line_delta: float, *, threshold: float = 70.0) -> int:
    """Return ``1`` when heavy public action and negative line movement align."""

    return int(ticket_percent >= threshold and line_delta < 0)


def reverse_line_move_flag(ticket_percent: float, line_delta: float, *, pivot: float = 50.0) -> int:
    """Return ``1`` when the line moves opposite the public betting split.

    ``ticket_percent`` represents the percentage of tickets on the team for this
    row. ``line_delta`` should be ``opening_odds - current_odds`` where positive
    values mean the price shortened. A reverse line move occurs when the market
    shifts toward the side receiving fewer tickets or drifts away from the heavy
    public side.
    """

    if pd.isna(ticket_percent) or pd.isna(line_delta):
        return 0
    if ticket_percent < pivot and line_delta > 0:
        return 1
    if ticket_percent > pivot and line_delta < 0:
        return 1
    return 0


def bet_freeze_flag(
    ticket_percent: float,
    line_delta: float,
    *,
    ticket_threshold: float = 70.0,
    move_threshold: float = LINE_MOVEMENT_THRESHOLD,
) -> int:
    """Return ``1`` when heavy public action and positive line movement align.

    ``ticket_percent`` is the percentage of bets on the team. ``line_delta``
    represents ``opening_odds - closing_odds``. A positive ``line_delta``
    indicates the odds shortened (e.g. ``-120`` to ``-140``). When a majority of
    tickets combine with a significant shorten in the same direction, sports
    books may be luring action on a trendy side, so this flag instructs the
    model to pause betting activity.
    """

    if pd.isna(ticket_percent) or pd.isna(line_delta):
        return 0
    return int(ticket_percent >= ticket_threshold and line_delta >= move_threshold)


def anti_correlation_flag(
    ticket_percent: float,
    line_delta: float,
    *,
    threshold: float = 70.0,
) -> int:
    """Return ``1`` when heavy public action meets a reverse line move.

    This flag captures a potential trap scenario where the majority of
    bets land on one team while the price drifts against them. When the
    public heavily favours a side (``ticket_percent`` at or above
    ``threshold``) yet the line moves the other way (``line_delta`` is
    negative), oddsmakers may be inviting action on a loser. The feature
    helps the model fade such spots.
    """

    if pd.isna(ticket_percent) or pd.isna(line_delta):
        return 0
    return int(ticket_percent >= threshold and line_delta < 0)


def bullpen_era_vs_opponent_slg(bullpen_era: float, opponent_slg: float) -> float:
    """Return bullpen ERA minus opponent slugging percentage.

    A positive value indicates a weaker bullpen relative to the power of the
    opposing lineup. ``opponent_slg`` can represent season-long or recent form
    depending on the dataset. Missing values yield ``NaN``.
    """

    if pd.isna(bullpen_era) or pd.isna(opponent_slg):
        return float("nan")
    return float(bullpen_era) - float(opponent_slg)


def attach_recency_weighted_features(
    df: pd.DataFrame,
    *,
    multiplier: float = 0.7,
    verbose: bool = False,
) -> None:
    """Add columns blending recent metrics with full-season stats.

    Any column named like ``<stat>_last_<N>`` will be combined with a matching
    ``<stat>`` column when present. The resulting ``<stat>_weighted_recent``
    column gives ``multiplier`` weight to the recent value and ``1 - multiplier``
    to the season-long value so the model can emphasize current form.
    """

    import re

    pattern = re.compile(r"(.+)_last_\d+(?:_games?|_starts?)?")
    added: list[str] = []
    for col in list(df.columns):
        m = pattern.match(col)
        if not m:
            continue
        base = m.group(1)
        if base in df.columns:
            out_col = f"{base}_weighted_recent"
            df[out_col] = (
                df[col] * multiplier + df[base] * (1.0 - multiplier)
            )
            added.append(out_col)
    if verbose and added:
        print(f"Computed recency weighted features: {', '.join(added)}")


def llm_sharp_context_score(text: str) -> float:
    """Return a score between 0 and 1 from OpenAI about sharp betting context."""
    if openai is None or not OPENAI_API_KEY:
        return 0.0
    prompt = (
        "Given the following context from news or social media, rate the likelihood "
        "that professional sharp bettors are influencing the market on a 0-1 scale:\n"
        f"{text}\nScore:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = resp.choices[0].message["content"].strip()
        return float(reply)
    except Exception:
        return 0.0


def _fetch_reddit_posts(subreddit: str, query: str, limit: int = 50) -> list[str]:
    """Return recent Reddit submissions matching ``query``."""
    if praw is None:
        return []
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "odds-fetcher")
    if not client_id or not client_secret:
        return []
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )
        posts = []
        for sub in reddit.subreddit(subreddit).search(query, limit=limit):
            posts.append(f"{sub.title}\n{sub.selftext}")
        return posts
    except Exception:
        return []


def _fetch_twitter_posts(query: str, limit: int = 50) -> list[str]:
    """Return recent tweets containing ``query``."""
    if tweepy is None:
        return []
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        return []
    try:
        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
        posts = []
        for tweet in tweepy.Paginator(
            client.search_recent_tweets, query=query, max_results=100
        ).flatten(limit=limit):
            if hasattr(tweet, "text"):
                posts.append(tweet.text)
        return posts
    except Exception:
        return []


def _fetch_telegram_messages(channel: str, limit: int = 50) -> list[str]:
    """Return recent messages from a Telegram channel."""
    if TelegramClient is None:
        return []
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")
    session_name = os.getenv("TG_SESSION", "odds_fetcher")
    if not api_id or not api_hash or not channel:
        return []
    client = TelegramClient(session_name, int(api_id), api_hash)
    posts: list[str] = []
    try:
        with client:
            for msg in client.iter_messages(channel, limit=limit):
                if msg.text:
                    posts.append(msg.text)
    except Exception:
        return posts
    return posts


def gather_social_text(team: str, limit: int = 50) -> list[str]:
    """Collect posts from Reddit, Twitter and Telegram about ``team``."""
    posts = []
    posts += _fetch_reddit_posts("sportsbook", team, limit=limit)
    posts += _fetch_twitter_posts(team, limit=limit)
    tg_channel = os.getenv("TG_CHANNEL")
    if tg_channel:
        posts += _fetch_telegram_messages(tg_channel, limit=limit)
    return posts


def llm_sharp_social_score(team: str, limit: int = 50) -> float:
    """Return a sharp money score from recent social chatter about ``team``."""
    texts = gather_social_text(team, limit=limit)
    if not texts:
        return 0.0
    combined = "\n".join(texts)
    return llm_sharp_context_score(combined[:4000])


def llm_managerial_signals(text: str) -> dict[str, int]:
    """Return managerial move flags extracted from commentary text."""
    # When OpenAI is unavailable simply fall back to a basic heuristic
    if openai is None or not OPENAI_API_KEY:
        early_pull = int("pulled" in text.lower())
        pinch_hit = int("pinch hit" in text.lower() or "pinch-hitter" in text.lower())
        matchup_move = int("matchup" in text.lower())
        return {
            "early_pull_flag": early_pull,
            "pinch_hit_flag": pinch_hit,
            "matchup_move_flag": matchup_move,
        }

    prompt = (
        "Identify whether the following commentary text mentions an early pull, "
        "a pinch-hitting decision or a matchup based pitching change. "
        "Respond with JSON like {\"early_pull_flag\":0,\"pinch_hit_flag\":0,"
        "\"matchup_move_flag\":0}.\nText:\n" + text + "\nJSON:" 
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = resp.choices[0].message["content"].strip()
        return json.loads(reply)
    except Exception:
        return {
            "early_pull_flag": 0,
            "pinch_hit_flag": 0,
            "matchup_move_flag": 0,
        }


def llm_lineup_risk_score(text: str) -> float:
    """Return lineup risk score from injury-related commentary text."""
    if openai is None or not OPENAI_API_KEY:
        return 0.0
    prompt = (
        "Assess the risk of last-minute lineup changes due to injuries. "
        "Look for phrases like 'late scratch', 'day-to-day' or "
        "'questionable start' and output a number from 0 to 1 where higher "
        "values indicate greater uncertainty.\nText:\n" + text + "\nScore:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = resp.choices[0].message["content"].strip()
        return float(reply)
    except Exception:
        return 0.0


def llm_lineup_risk_social_score(team: str, limit: int = 50) -> float:
    """Return lineup risk score from recent social chatter about ``team``."""
    texts = gather_social_text(team, limit=limit)
    if not texts:
        return 0.0
    combined = "\n".join(texts)
    return llm_lineup_risk_score(combined[:4000])


def llm_hype_trend_score(text: str) -> float:
    """Return a score between 0 and 1 quantifying hype and overexcitement."""
    if openai is None or not OPENAI_API_KEY:
        return 0.0
    prompt = (
        "Evaluate the following social chatter for excessive hype or viral "
        "narratives about a team. Look for sudden surges in mentions or "
        "statements like 'can't lose today'. Output a number from 0 to 1 "
        "where 1 means the crowd is extremely overconfident.\nText:\n"
        + text
        + "\nScore:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = resp.choices[0].message["content"].strip()
        return float(reply)
    except Exception:
        return 0.0


def llm_hype_trend_social_score(team: str, limit: int = 50) -> float:
    """Return hype trend score from recent social chatter about ``team``."""
    texts = gather_social_text(team, limit=limit)
    if not texts:
        return 0.0
    combined = "\n".join(texts)
    return llm_hype_trend_score(combined[:4000])


def llm_sentiment_fakeout_score(text: str) -> float:
    """Return a score between 0 and 1 for sarcastic or misleading sentiment."""
    if openai is None or not OPENAI_API_KEY:
        return 0.0
    prompt = (
        "Analyze the following messages for sarcasm, ironic hype or "
        "other sentiment reversals that might mislead crowd emotion. "
        "Output a number from 0 to 1 where higher values mean the "
        "apparent sentiment is likely a fake-out.\nText:\n"
        + text
        + "\nScore:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = resp.choices[0].message["content"].strip()
        return float(reply)
    except Exception:
        return 0.0


def llm_sentiment_fakeout_social_score(team: str, limit: int = 50) -> float:
    """Return fakeout sentiment score from social chatter about ``team``."""
    texts = gather_social_text(team, limit=limit)
    if not texts:
        return 0.0
    combined = "\n".join(texts)
    return llm_sentiment_fakeout_score(combined[:4000])


def llm_inning_trend_summaries(text: str) -> dict[int, str]:
    """Return concise inning summaries extracted from raw play-by-play logs."""

    if openai is None or not OPENAI_API_KEY:
        return {}

    prompt = (
        "Summarize the following baseball game log with one short sentence per "
        "inning. Format each line as 'Inning X: <summary>'.\n\n" + text + "\n\nSummaries:"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = resp.choices[0].message["content"].strip()
    except Exception:
        return {}

    import re

    summaries: dict[int, str] = {}
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(?:Inning\s*)?(\d+)[:.]\s*(.+)", line)
        if m:
            inning = int(m.group(1))
            summaries[inning] = m.group(2).strip()

    return summaries


def attach_managerial_signals(
    df: pd.DataFrame,
    column: str,
    *,
    verbose: bool = False,
) -> None:
    """Add managerial decision flags derived from commentary column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    flags = df[column].fillna("").apply(llm_managerial_signals)
    df["early_pull_flag"] = flags.apply(lambda d: d["early_pull_flag"])
    df["pinch_hit_flag"] = flags.apply(lambda d: d["pinch_hit_flag"])
    df["matchup_move_flag"] = flags.apply(lambda d: d["matchup_move_flag"])
    if verbose:
        print(f"Computed managerial signals using column '{column}'")


def attach_lineup_risk_scores(
    df: pd.DataFrame,
    team_column: str,
    *,
    limit: int = 50,
    verbose: bool = False,
) -> None:
    """Add ``lineup_risk_score`` column based on a team name column."""
    if team_column not in df.columns:
        raise ValueError(f"Column '{team_column}' not found in dataframe")
    df["lineup_risk_score"] = df[team_column].apply(
        lambda t: llm_lineup_risk_social_score(str(t), limit=limit)
    )
    if verbose:
        print(f"Computed lineup_risk_score using column '{team_column}'")


def attach_social_scores(
    df: pd.DataFrame,
    team_column: str,
    *,
    limit: int = 50,
    verbose: bool = False,
) -> None:
    """Add ``sharp_money_score_social`` column based on a team name column."""
    if team_column not in df.columns:
        raise ValueError(f"Column '{team_column}' not found in dataframe")
    df["sharp_money_score_social"] = df[team_column].apply(
        lambda t: llm_sharp_social_score(str(t), limit=limit)
    )
    if verbose:
        print(f"Computed sharp_money_score_social using column '{team_column}'")


def attach_hype_trend_scores(
    df: pd.DataFrame,
    team_column: str,
    *,
    limit: int = 50,
    verbose: bool = False,
) -> None:
    """Add ``hype_trend_score`` column based on a team name column."""
    if team_column not in df.columns:
        raise ValueError(f"Column '{team_column}' not found in dataframe")
    df["hype_trend_score"] = df[team_column].apply(
        lambda t: llm_hype_trend_social_score(str(t), limit=limit)
    )
    if verbose:
        print(f"Computed hype_trend_score using column '{team_column}'")


def attach_sentiment_fakeout_scores(
    df: pd.DataFrame,
    team_column: str,
    *,
    limit: int = 50,
    verbose: bool = False,
) -> None:
    """Add ``sentiment_fakeout_score`` column based on a team name column."""
    if team_column not in df.columns:
        raise ValueError(f"Column '{team_column}' not found in dataframe")
    df["sentiment_fakeout_score"] = df[team_column].apply(
        lambda t: llm_sentiment_fakeout_social_score(str(t), limit=limit)
    )
    if verbose:
        print(f"Computed sentiment_fakeout_score using column '{team_column}'")

ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("THE_ODDS_API_KEY environment variable is not set (check your .env file)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if openai is not None:
    openai.api_key = OPENAI_API_KEY
elif OPENAI_API_KEY:
    try:  # Dynamically import openai if API key is present
        import openai as _openai
        openai = _openai  # type: ignore
        openai.api_key = OPENAI_API_KEY
    except Exception:  # pragma: no cover - optional dependency
        openai = None

MAX_HISTORICAL_DAYS = 365

H2H_DATA_DIR = ROOT_DIR / "h2h_data"
H2H_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = H2H_DATA_DIR / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
H2H_MODEL_PATH = H2H_DATA_DIR / "h2h_classifier.pkl"
MONEYLINE_MODEL_PATH = ROOT_DIR / "moneyline_classifier.pkl"
DUAL_HEAD_MODEL_PATH = ROOT_DIR / "moneyline_dual_head.pkl"
MARKET_MAKER_MIRROR_MODEL_PATH = ROOT_DIR / "market_maker_mirror.pkl"


def sanitize_path(path: str) -> str:
    """Return an absolute normalized form of ``path``."""
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    return str(Path(path).resolve(strict=False))


def _coerce_object_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with object columns converted to numeric when possible."""
    df = pd.DataFrame(df)
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def to_fixed_utc(date_obj: datetime) -> str:
    """Return ISO-8601 string at fixed 12:00 UTC."""
    return date_obj.strftime("%Y-%m-%dT12:00:00Z")

def _safe_cache_key(*args) -> str:
    str_key = "-".join(str(x) for x in args)
    return hashlib.md5(str_key.encode()).hexdigest()

def _cache_load(cache_dir: Path, key: str):
    cache_path = cache_dir / f"{key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def _cache_save(cache_dir: Path, key: str, data) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

def build_event_ids_url(
    sport_key: str,
    date: str,
    regions: str = "us",
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "h2h",
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_event_ids_historical(
    sport_key: str,
    date: str,
    regions: str = "us",
) -> list:
    """Fetch all event IDs for a given sport and date (historical snapshot)."""
    url = build_event_ids_url(sport_key, date, regions)
    cache_key = _safe_cache_key("eventids", sport_key, date, regions, "h2h")
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, list):
            raise ValueError(f"Unexpected event ids response: {data!r}")
        event_ids = []
        for g in data:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            for book in g.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == "h2h":
                        event_ids.append(g["id"])
                        break
                else:
                    continue
                break
        _cache_save(CACHE_DIR, cache_key, event_ids)
        return event_ids
    except Exception as e:
        print(f"Error fetching event ids for {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_h2h_event_ids_url(
    sport_key: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
) -> str:
    """Build URL for head-to-head event IDs."""
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_h2h_event_ids(
    sport_key: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
    verbose: bool = False,
) -> list:
    """Fetch event IDs using the provided API key."""
    url = build_h2h_event_ids_url(
        sport_key, date, api_key=api_key, regions=regions
    )
    cache_key = _safe_cache_key(
        "h2h_event_ids", sport_key, date, regions, api_key
    )
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, list):
            raise ValueError(f"Unexpected event ids response: {data!r}")
        event_ids = []
        for g in data:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            for book in g.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == "h2h":
                        event_ids.append(g["id"])
                        break
                else:
                    continue
                break
        _cache_save(CACHE_DIR, cache_key, event_ids)
        if verbose:
            print(f"Fetched {len(event_ids)} event ids for {date}")
        return event_ids
    except Exception as e:
        print(f"Error fetching h2h event ids for {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_historical_odds_url(
    sport_key: str,
    date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "h2h",
        "date": date,
        "oddsFormat": odds_format,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_historical_h2h_odds(
    sport_key: str,
    date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
) -> list:
    """Return all h2h odds for a sport and date."""
    url = build_historical_odds_url(
        sport_key, date, regions=regions, odds_format=odds_format
    )
    cache_key = _safe_cache_key(
        "historicalodds", sport_key, date, regions, odds_format
    )
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())

        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "data" in data:
            events = data["data"]
            if not isinstance(events, list):
                raise ValueError(
                    f"Unexpected historical odds response: {data!r}"
                )
        else:
            raise ValueError(f"Unexpected historical odds response: {data!r}")

        _cache_save(CACHE_DIR, cache_key, events)
        return events
    except Exception as e:
        print(f"Error fetching historical odds for {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []

def build_h2h_url_historical(
    sport_key: str,
    event_id: str,
    date: str,
    regions: str = "us",
    date_format: str = "iso",
    odds_format: str = "american",
) -> str:
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    )
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_h2h_props_historical(
    sport_key: str,
    event_id: str,
    date: str,
    regions: str = "us",
    odds_format: str = "american",
) -> list:
    cache_key = _safe_cache_key("h2hprops", sport_key, event_id, date, regions, odds_format)
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = build_h2h_url_historical(
        sport_key, event_id, date=date, regions=regions, odds_format=odds_format
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())

            out: list
            if isinstance(data, dict):
                if "bookmakers" in data:
                    out = data["bookmakers"]
                elif "data" in data and isinstance(data["data"], dict):
                    out = data["data"].get("bookmakers", [])
                else:
                    out = []
            else:
                out = []

            _cache_save(CACHE_DIR, cache_key, out)
            return out
    except Exception as e:
        print(f"Error fetching h2h props for event {event_id} on {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []


def build_h2h_props_url(
    sport_key: str,
    event_id: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
    date_format: str = "iso",
    odds_format: str = "american",
) -> str:
    """Build URL for head-to-head props."""
    base_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    )
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "date": date,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_h2h_props(
    sport_key: str,
    event_id: str,
    date: str,
    *,
    api_key: str,
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> list:
    """Fetch h2h props for the given event ID."""
    cache_key = _safe_cache_key(
        "h2h_props", sport_key, event_id, date, regions, odds_format, api_key
    )
    cached = _cache_load(CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = build_h2h_props_url(
        sport_key,
        event_id,
        date,
        api_key=api_key,
        regions=regions,
        odds_format=odds_format,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, dict) and "bookmakers" in data:
                out = data["bookmakers"]
            else:
                out = []
            _cache_save(CACHE_DIR, cache_key, out)
            if verbose:
                print(
                    f"Fetched props for event {event_id} on {date} ({len(out)} bookmakers)"
                )
            return out
    except Exception as e:
        print(f"Error fetching h2h props for event {event_id} on {date}: {e}")
        _cache_save(CACHE_DIR, cache_key, [])
        return []

def build_h2h_dataset_from_api(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
) -> pd.DataFrame:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    # The Odds API only keeps about one year of historical results. When the
    # requested window falls outside that range the API returns no data,
    # resulting in a confusing "No h2h data returned" error. Clamp the start
    # date to the earliest supported day and warn the user if needed.
    window_start = datetime.utcnow() - timedelta(days=365)
    if start < window_start:
        if verbose:
            print(
                f"Start date {start_date} is before the API window. "
                f"Using {window_start.date()} instead."
            )
        start = window_start

    if end < start:
        raise ValueError("end_date must be on or after start_date")
    rows: list[dict] = []
    current = start
    while current <= end:
        date_str = to_fixed_utc(current)
        events = fetch_historical_h2h_odds(
            sport_key,
            date_str,
            regions=regions,
            odds_format=odds_format,
        )
        if verbose:
            print(f"Fetched {len(events)} events for {date_str}")
        for game in events:
            if not isinstance(game, dict):
                continue
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) != 2:
                        continue
                    team1 = outcomes[0].get("name")
                    team2 = outcomes[1].get("name")
                    price1 = outcomes[0].get("price")
                    price2 = outcomes[1].get("price")
                    result1 = outcomes[0].get("result")
                    result2 = outcomes[1].get("result")
                    if None in (team1, team2, price1, price2, result1, result2):
                        continue
                    if verbose:
                        print(
                            f"DEBUG: {team1} vs {team2} | price1={price1}, price2={price2}, result1={result1}, result2={result2}"
                        )
                    label = 1 if result1 == "win" else 0
                    rows.append({
                        "team1": team1,
                        "team2": team2,
                        "price1": price1,
                        "price2": price2,
                        "implied_prob": american_odds_to_prob(price1),
                        "team1_win": label,
                        "event_date": current.date().isoformat(),
                    })
                    break
        current += timedelta(days=1)
    if not rows:
        raise RuntimeError("No h2h data returned")
    if verbose:
        print(f"Built h2h dataset with {len(rows)} rows.")
    return pd.DataFrame(rows)

class SegmentedCalibratedModel:
    """Logistic regression with per-inning probability calibration."""

    def __init__(self, pipeline, calibrators: dict[str, CalibratedClassifierCV]):
        self.pipeline = pipeline
        self.calibrators = calibrators

    def _segment_masks(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        seg7 = X.get(
            "live_inning_7_diff",
            pd.Series([np.nan] * len(X), index=X.index),
        ).notna()
        seg5 = (
            X.get(
                "live_inning_5_diff",
                pd.Series([np.nan] * len(X), index=X.index),
            ).notna()
            & ~seg7
        )
        seg_pre = ~seg5 & ~seg7
        return {"7th_inning": seg7, "5th_inning": seg5, "pregame": seg_pre}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame(X)
        proba = np.zeros(len(df))
        masks = self._segment_masks(df)
        for seg, mask in masks.items():
            if not mask.any():
                continue
            calibrator = self.calibrators.get(seg)
            if calibrator is not None:
                proba[mask] = calibrator.predict_proba(df[mask])[:, 1]
            else:
                proba[mask] = self.pipeline.predict_proba(df[mask])[:, 1]
        return np.column_stack([1 - proba, proba])


def _train(
    X: pd.DataFrame,
    y: pd.Series,
    model_out: str,
    sample_weight: pd.Series | None = None,
    *,
    profile_memory: bool = False,
) -> None:
    """Train a logistic regression model with segment-specific calibration."""

    model_out = sanitize_path(model_out)

    # Ensure training data only contains numeric columns. When reading from CSV
    # pandas may interpret numeric fields as strings if the dataset contains
    # mixed types or missing values. Attempt to coerce such object columns to
    # numeric before dropping non-numeric data so valid training rows are not
    # discarded inadvertently.
    X = pd.DataFrame(X)
    obj_cols = X.select_dtypes(include="object").columns
    for col in obj_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.select_dtypes(include=[np.number, bool]).fillna(0)

    # Remove rows where the target is missing
    mask = pd.Series(y).notna()
    if mask.sum() != len(y):
        X = X.loc[mask]
        y = pd.Series(y)[mask]
        if sample_weight is not None:
            sample_weight = pd.Series(sample_weight)[mask]

    if len(X) == 0:
        raise ValueError(
            "Training data is empty after preprocessing. "
            "Check that the dataset contains numeric feature values and non-missing labels."
        )

    ctx = memory_usage("train_split") if profile_memory else nullcontext()
    with ctx:
        if sample_weight is not None:
            X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                X, y, sample_weight, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    ctx = memory_usage("fit_model") if profile_memory else nullcontext()
    with ctx:
        if sample_weight is not None:
            pipeline.fit(X_train, y_train, sample_weight=w_train)
        else:
            pipeline.fit(X_train, y_train)

    # Build segment specific calibrators
    masks = {
        "7th_inning": X_val.get(
            "live_inning_7_diff",
            pd.Series([np.nan] * len(X_val), index=X_val.index),
        ).notna(),
    }
    masks["5th_inning"] = (
        X_val.get(
            "live_inning_5_diff",
            pd.Series([np.nan] * len(X_val), index=X_val.index),
        ).notna()
        & ~masks["7th_inning"]
    )
    masks["pregame"] = ~masks["5th_inning"] & ~masks["7th_inning"]

    calibrators: dict[str, CalibratedClassifierCV] = {}
    for seg, mask in masks.items():
        if not mask.any():
            continue
        cal = CalibratedClassifierCV(
            FrozenEstimator(pipeline),
            method="isotonic",
        )
        if sample_weight is not None:
            cal.fit(X_val[mask], y_val[mask], sample_weight=w_val[mask])
        else:
            cal.fit(X_val[mask], y_val[mask])
        calibrators[seg] = cal

    model = SegmentedCalibratedModel(pipeline, calibrators)

    probas = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probas)
    brier = brier_score_loss(y_val, probas)
    print(f"Validation AUC: {auc:.3f}, Brier score: {brier:.3f}")

    def _report_segment_metrics() -> None:
        for seg, mask in masks.items():
            if not mask.any() or seg not in calibrators:
                continue
            seg_y = y_val.loc[mask]
            seg_proba = calibrators[seg].predict_proba(X_val[mask])[:, 1]
            seg_auc = roc_auc_score(seg_y, seg_proba)
            seg_brier = brier_score_loss(seg_y, seg_proba)
            name = "7th inning" if seg == "7th_inning" else ("5th inning" if seg == "5th_inning" else "pregame")
            print(f"  {name} AUC: {seg_auc:.3f}, Brier: {seg_brier:.3f}")

    if masks:
        _report_segment_metrics()

    residuals_df = pd.DataFrame({
        "true_label": y_val.reset_index(drop=True),
        "model_prob": probas,
    })
    residuals_df["residual"] = residuals_df["true_label"] - residuals_df["model_prob"]

    residuals_path = Path(model_out).with_suffix(".residuals.csv")
    residuals_df.to_csv(residuals_path, index=False)
    print(f"Residuals saved to {residuals_path}")

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # Persist the training columns alongside the model so inference can
        # easily align feature order.
        pickle.dump((model, list(X.columns)), f)

    print(f"Model saved to {out_path}")
    return model

def train_h2h_classifier(
    sport_key: str,
    start_date: str,
    end_date: str,
    *,
    model_out: str = str(H2H_MODEL_PATH),
    regions: str = "us",
    odds_format: str = "american",
    verbose: bool = False,
    recent_half_life: float | None = None,
    profile_memory: bool = False,
) -> None:
    with memory_usage("build_dataset") if profile_memory else nullcontext():
        df = build_h2h_dataset_from_api(
            sport_key,
            start_date,
            end_date,
            regions=regions,
            odds_format=odds_format,
            verbose=verbose,
        )
    if verbose:
        print(df.head())
    X = df[["price1", "price2"]]
    y = df["team1_win"]
    weights = None
    if recent_half_life is not None and "event_date" in df.columns:
        weights = compute_recency_weights(df["event_date"], half_life_days=recent_half_life)
    with memory_usage("train") if profile_memory else nullcontext():
        _train(
            X,
            y,
            sanitize_path(model_out),
            sample_weight=weights,
            profile_memory=profile_memory,
        )

def predict_h2h_probability(
    model_path: str,
    price1: float,
    price2: float,
) -> float:
    model_path = sanitize_path(model_path)
    with open(model_path, "rb") as f:
        try:
            model_info = pickle.load(f)
        except AttributeError as exc:
            raise RuntimeError(
                "Invalid or outdated model file. Train a new classifier using 'python3 main.py train_classifier'."
            ) from exc
    if isinstance(model_info, tuple):
        model, cols = model_info
    else:
        model = model_info
        cols = None

    df = pd.DataFrame([{"price1": price1, "price2": price2}])
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"Missing feature columns: {', '.join(missing)}",
                RuntimeWarning,
            )
        df = df.reindex(cols, axis=1, fill_value=0)

    proba = model.predict_proba(df)[0][1]
    return float(proba)


class DualHeadModel:
    """Model containing separate pregame and live heads."""

    def __init__(
        self,
        pregame_model: SegmentedCalibratedModel,
        live_model: SegmentedCalibratedModel,
        pregame_cols: list[str],
        live_cols: list[str],
    ) -> None:
        self.pregame_model = pregame_model
        self.live_model = live_model
        self.pregame_cols = pregame_cols
        self.live_cols = live_cols

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame(X)
        if set(self.live_cols).issubset(df.columns) and df[self.live_cols].notna().any().any():
            proba = self.live_model.predict_proba(df[self.live_cols])[:, 1]
        else:
            proba = self.pregame_model.predict_proba(df[self.pregame_cols])[:, 1]
        return np.column_stack([1 - proba, proba])


def split_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate pregame and live-game feature DataFrames.

    Any columns containing post-result information are discarded to avoid data
    leakage when training or predicting.
    """

    disallow_keywords = {"result", "final", "post", "future"}

    def allowed(col: str) -> bool:
        c_lower = col.lower()
        return not any(k in c_lower for k in disallow_keywords)

    pregame_cols = [c for c in df.columns if c.startswith("pregame_") and allowed(c)]
    live_cols = [c for c in df.columns if c.startswith("live_") and allowed(c)]

    # Ignore any columns that might leak future information
    other_cols = [
        c
        for c in df.columns
        if c not in pregame_cols + live_cols and allowed(c)
    ]

    pregame_df = df[pregame_cols + other_cols].copy()
    live_df = df[live_cols].copy()
    return pregame_df, live_df


def train_moneyline_classifier(
    dataset_path: str,
    *,
    model_out: str = str(MONEYLINE_MODEL_PATH),
    features_type: str = "pregame",
    verbose: bool = False,
    recent_half_life: float | None = None,
    date_column: str | None = None,
    recency_multiplier: float = 0.7,
    profile_memory: bool = False,
) -> None:
    """Train a logistic regression model from a CSV with a home_team_win column.

Modeling is done in regression mode first. You can later apply a probability threshold for classification if desired."""
    dataset_path = sanitize_path(dataset_path)
    model_out = sanitize_path(model_out)
    with memory_usage("read_csv") if profile_memory else nullcontext():
        df = pd.read_csv(dataset_path)
    if "home_team_win" not in df.columns:
        # Older datasets produced by data_prep.py used 'team1_win'
        if "team1_win" in df.columns:
            df = df.rename(columns={"team1_win": "home_team_win"})
        else:
            raise ValueError("Dataset must include 'home_team_win' column")

    # Automatically create line movement features when possible
    if {"opening_odds", "closing_odds"}.issubset(df.columns):
        df["line_delta"] = line_movement_delta(df["opening_odds"], df["closing_odds"])
        df["line_movement_delta"] = df["line_delta"].apply(line_movement_flag)
        if verbose:
            print(
                "Computed line_delta and line_movement_delta from opening and closing odds"
            )

    # Create sharp money features when handle/ticket percentages are available
    if {"handle_percent", "ticket_percent"}.issubset(df.columns):
        delta = df["handle_percent"] - df["ticket_percent"]
        df["sharp_money_delta"] = delta
        df["sharp_action_flag"] = (delta > 20).astype(int)
        df["sharp_money_score"] = delta / 100.0
        if verbose:
            print(
                "Computed sharp_money_delta, sharp_action_flag and sharp_money_score from handle/ticket percentages"
            )

    # Public betting fade signal when ticket percentage shows heavy bias
    if {"ticket_percent", "line_delta"}.issubset(df.columns):
        df["public_fade_flag"] = df.apply(
            lambda r: public_fade_flag(r["ticket_percent"], r["line_delta"]), axis=1
        )
        df["reverse_line_move"] = df.apply(
            lambda r: reverse_line_move_flag(r["ticket_percent"], r["line_delta"]),
            axis=1,
        )
        df["bet_freeze_flag"] = df.apply(
            lambda r: bet_freeze_flag(r["ticket_percent"], r["line_delta"]),
            axis=1,
        )
        df["anti_correlation_flag"] = df.apply(
            lambda r: anti_correlation_flag(r["ticket_percent"], r["line_delta"]),
            axis=1,
        )
        if verbose:
            print(
                "Computed public_fade_flag, reverse_line_move, bet_freeze_flag and anti_correlation_flag from ticket_percent and line_delta"
            )

    # Generate LLM-based sharp context score if a context column exists
    context_col = next((c for c in df.columns if "context" in c.lower()), None)
    if context_col is not None:
        df["sharp_context_score"] = df[context_col].apply(llm_sharp_context_score)
        if verbose:
            print(f"Computed sharp_context_score using OpenAI on column '{context_col}'")

    commentary_col = next(
        (
            c
            for c in df.columns
            if any(k in c.lower() for k in ["manager", "commentary", "summary", "notes"])
        ),
        None,
    )
    if commentary_col is not None:
        _flags = df[commentary_col].fillna("").apply(llm_managerial_signals)
        df["early_pull_flag"] = _flags.apply(lambda d: d["early_pull_flag"])
        df["pinch_hit_flag"] = _flags.apply(lambda d: d["pinch_hit_flag"])
        df["matchup_move_flag"] = _flags.apply(lambda d: d["matchup_move_flag"])
        if verbose:
            print(
                f"Computed managerial signals using OpenAI on column '{commentary_col}'"
            )

    bullpen_col = next((c for c in df.columns if "bullpen_era" in c.lower()), None)
    slg_col = next((c for c in df.columns if "opponent_slg" in c.lower()), None)
    if bullpen_col and slg_col:
        df["bullpenERA_vs_opponentSLG"] = df.apply(
            lambda r: bullpen_era_vs_opponent_slg(r[bullpen_col], r[slg_col]), axis=1
        )
        if verbose:
            print(
                f"Computed bullpenERA_vs_opponentSLG using columns '{bullpen_col}' and '{slg_col}'"
            )

    # Blend recent form metrics with season-long stats
    attach_recency_weighted_features(df, multiplier=recency_multiplier, verbose=verbose)

    features_df = df.drop(columns=["home_team_win"])
    features_df = _coerce_object_numeric(features_df)
    features_df = features_df.select_dtypes(include=[np.number, bool]).fillna(0)
    pregame_X, live_X = split_feature_sets(features_df)
    X = live_X if features_type == "live" else pregame_X
    y = df["home_team_win"]
    if verbose:
        print(
            f"Training dataset with {len(df)} rows using {features_type} features ({len(X.columns)} columns)"
        )

    if X.empty:
        raise ValueError(f"No columns found for feature type: {features_type}")

    weights = None
    if recent_half_life is not None:
        col = (
            date_column
            if date_column and date_column in df.columns
            else next((c for c in df.columns if "date" in c.lower()), None)
        )
        if col is not None:
            weights = compute_recency_weights(
                pd.to_datetime(df[col], errors="coerce"),
                half_life_days=recent_half_life,
            )
            if verbose:
                print(f"Using column '{col}' for recency weighting")

    with memory_usage("train") if profile_memory else nullcontext():
        _train(
            X,
            y,
            model_out,
            sample_weight=weights,
            profile_memory=profile_memory,
        )


def train_dual_head_classifier(
    dataset_path: str,
    *,
    model_out: str = str(DUAL_HEAD_MODEL_PATH),
    verbose: bool = False,
    recent_half_life: float | None = None,
    date_column: str | None = None,
    recency_multiplier: float = 0.7,
    profile_memory: bool = False,
) -> None:
    """Train separate pregame and live models and save a ``DualHeadModel``."""

    dataset_path = sanitize_path(dataset_path)
    model_out = sanitize_path(model_out)
    with memory_usage("read_csv") if profile_memory else nullcontext():
        df = pd.read_csv(dataset_path)
    if "home_team_win" not in df.columns:
        if "team1_win" in df.columns:
            df = df.rename(columns={"team1_win": "home_team_win"})
        else:
            raise ValueError("Dataset must include 'home_team_win' column")

    attach_recency_weighted_features(df, multiplier=recency_multiplier, verbose=verbose)

    features_df = df.drop(columns=["home_team_win"])
    features_df = _coerce_object_numeric(features_df)
    features_df = features_df.select_dtypes(include=[np.number, bool]).fillna(0)
    pregame_X, live_X = split_feature_sets(features_df)
    y = df["home_team_win"]
    if verbose:
        print(
            f"Training dual-head model with {len(pregame_X.columns)} pregame and {len(live_X.columns)} live features"
        )

    weights = None
    if recent_half_life is not None:
        col = (
            date_column
            if date_column and date_column in df.columns
            else next((c for c in df.columns if "date" in c.lower()), None)
        )
        if col is not None:
            weights = compute_recency_weights(
                pd.to_datetime(df[col], errors="coerce"),
                half_life_days=recent_half_life,
            )
            if verbose:
                print(f"Using column '{col}' for recency weighting")

    # Train each head using temporary paths to avoid saving intermediate files
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp_pre, tempfile.NamedTemporaryFile(delete=False) as tmp_live:
        with memory_usage("train_pregame") if profile_memory else nullcontext():
            pre_model = _train(
                pregame_X,
                y,
                tmp_pre.name,
                sample_weight=weights,
                profile_memory=profile_memory,
            )
        with memory_usage("train_live") if profile_memory else nullcontext():
            live_model = _train(
                live_X,
                y,
                tmp_live.name,
                sample_weight=weights,
                profile_memory=profile_memory,
            )

    for path in (tmp_pre.name, tmp_pre.name + ".residuals.csv", tmp_live.name, tmp_live.name + ".residuals.csv"):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    model = DualHeadModel(pre_model, live_model, list(pregame_X.columns), list(live_X.columns))

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"Dual-head model saved to {out_path}")


def train_market_maker_mirror_model(
    dataset_path: str,
    *,
    model_out: str = str(MARKET_MAKER_MIRROR_MODEL_PATH),
    verbose: bool = False,
    profile_memory: bool = False,
) -> None:
    """Train a simple linear model that mirrors sharp bookmaker adjustments."""

    dataset_path = sanitize_path(dataset_path)
    model_out = sanitize_path(model_out)
    with memory_usage("read_csv") if profile_memory else nullcontext():
        df = pd.read_csv(dataset_path)
    required = [
        "opening_odds",
        "handle_percent",
        "ticket_percent",
        "volatility",
        "closing_odds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset must include columns: {', '.join(missing)}"
        )

    X = df[["opening_odds", "handle_percent", "ticket_percent", "volatility"]].fillna(0)
    y = df["closing_odds"]

    model = LinearRegression()
    with memory_usage("fit_model") if profile_memory else nullcontext():
        model.fit(X, y)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"Market maker mirror model saved to {out_path}")


def predict_market_maker_price(model_path: str, features: dict) -> float:
    """Predict the line a high efficiency book would offer."""
    model_path = sanitize_path(model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    cols = ["opening_odds", "handle_percent", "ticket_percent", "volatility"]
    df = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])
    price = model.predict(df)[0]
    return float(price)


def market_maker_mirror_score(
    model_path: str,
    features: dict,
    current_odds: float,
) -> float:
    """Return difference between mirrored implied probability and current odds."""
    try:
        mirror_price = predict_market_maker_price(model_path, features)
    except Exception:
        return 0.0
    implied_mirror = american_odds_to_prob(mirror_price)
    implied_current = american_odds_to_prob(current_odds)
    return implied_mirror - implied_current


def predict_moneyline_probability(
    model_path: str,
    features: dict,
) -> float:
    """Predict win probability using a trained moneyline classifier."""
    model_path = sanitize_path(model_path)
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)

    if isinstance(model_info, tuple):
        model, cols = model_info
    else:
        model = model_info
        cols = getattr(model, "pregame_cols", None)

    df = pd.DataFrame([features])
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"Missing feature columns: {', '.join(missing)}",
                RuntimeWarning,
            )
        df = df.reindex(cols, axis=1, fill_value=0)

    proba = model.predict_proba(df)[0][1]
    return float(proba)


def extract_advanced_ml_features(
    model_path: str,
    *,
    price1: float,
    price2: float,
    team1: str | None = None,
    team2: str | None = None,
) -> dict:
    """Return additional metrics from a moneyline or dual-head model."""

    base_features = {"price1": price1, "price2": price2}
    if team1 is not None:
        base_features["team1"] = team1
    if team2 is not None:
        base_features["team2"] = team2

    try:
        prob = predict_moneyline_probability(model_path, base_features)
    except Exception:
        return {}

    implied = american_odds_to_prob(price1)
    edge = prob - implied
    ev = edge * american_odds_to_payout(price1)
    return {
        "advanced_ml_prob": prob,
        "advanced_ml_edge": edge,
        "advanced_ml_ev": ev,
    }


def extract_market_signals(
    model_path: str,
    *,
    price1: float,
    ticket_percent: float,
) -> dict:
    """Return market maker mirror metrics for the given line."""

    features = {
        "opening_odds": price1,
        "handle_percent": ticket_percent,
        "ticket_percent": ticket_percent,
        "volatility": 0.0,
    }

    try:
        mirror_price = predict_market_maker_price(model_path, features)
        mirror_score = market_maker_mirror_score(model_path, features, price1)
    except Exception:
        return {}

    return {
        "predicted_mirror_price": mirror_price,
        "predicted_mirror_probability": american_odds_to_prob(mirror_price),
        "mirror_score": mirror_score,
    }

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train a head-to-head classifier using historical odds endpoint")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model-out", default=str(H2H_MODEL_PATH))
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    train_h2h_classifier(
        args.sport,
        args.start_date,
        args.end_date,
        model_out=args.model_out,
        verbose=args.verbose,
    )


def demo_fetch() -> None:
    """Example usage of fetching h2h data."""
    event_ids = fetch_h2h_event_ids(
        sport_key="baseball_mlb",
        date="2025-06-01T12:00:00Z",
        api_key=API_KEY,
        verbose=True,
    )

    for event_id in event_ids:
        bookmakers = fetch_h2h_props(
            sport_key="baseball_mlb",
            event_id=event_id,
            date="2025-06-01T12:00:00Z",
            api_key=API_KEY,
            verbose=True,
        )
        # process bookmakers as before...

if __name__ == "__main__":
    _cli()
    demo_fetch()
