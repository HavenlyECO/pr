import os
import time
import requests
import re
import pickle
from collections import defaultdict
from typing import Dict, List

# 1. Static public bias dictionary (example values, to be tuned with real data)
MLB_PUBLIC_BIAS = {
    "New York Yankees": 0.95,
    "Los Angeles Dodgers": 0.93,
    "Boston Red Sox": 0.91,
    "Chicago Cubs": 0.89,
    "Atlanta Braves": 0.87,
    "Houston Astros": 0.86,
    "Chicago White Sox": 0.84,
    "San Francisco Giants": 0.82,
    "St. Louis Cardinals": 0.81,
    "Philadelphia Phillies": 0.80,
    # ... all MLB teams ...
}

def public_bias_score(team_name: str) -> float:
    """Return static public bias score for an MLB team."""
    return MLB_PUBLIC_BIAS.get(team_name, 0.5)  # Default to 0.5 if unknown

# 2. Reddit sentiment feature extraction
REDDIT_CACHE_FILE = "reddit_sentiment_cache.pkl"
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sports-betting-toolkit/0.1")

def _load_reddit_cache():
    if os.path.exists(REDDIT_CACHE_FILE):
        with open(REDDIT_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def _save_reddit_cache(cache):
    with open(REDDIT_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def fetch_reddit_posts(team_name: str, limit=40, hours_back=12) -> List[str]:
    """Fetch recent Reddit posts mentioning the team from /r/baseball."""
    import praw
    cache = _load_reddit_cache()
    cache_key = f"{team_name}_{int(time.time() // 3600)}"
    if cache_key in cache:
        return cache[cache_key]
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    subreddit = reddit.subreddit("baseball")
    posts = []
    now = int(time.time())
    pattern = re.compile(re.escape(team_name), re.I)
    for submission in subreddit.new(limit=250):
        if (now - submission.created_utc) > hours_back * 3600:
            continue
        if pattern.search(submission.title) or pattern.search(submission.selftext):
            posts.append(submission.title + " " + submission.selftext)
            if len(posts) >= limit:
                break
    cache[cache_key] = posts
    _save_reddit_cache(cache)
    return posts

def _call_sentiment_llm(texts: List[str], mode: str) -> float:
    """Call LLM (OpenAI or open-source) for sentiment scoring (mock implementation)."""
    # Placeholder: In real usage, call OpenAI or a sentiment model here.
    # For now, return a mock score based on keywords.
    if not texts:
        return 0.0
    sharp_keywords = ["value", "underdog", "market", "edge", "price"]
    hype_keywords = ["lock", "must win", "easy", "guaranteed"]
    risk_keywords = ["injury", "questionable", "doubt", "uncertain", "out"]
    score = 0
    total = 0
    for t in texts:
        t = t.lower()
        if mode == "sharp":
            score += any(w in t for w in sharp_keywords)
        elif mode == "hype":
            score += any(w in t for w in hype_keywords)
        elif mode == "risk":
            score += any(w in t for w in risk_keywords)
        total += 1
    return score / total if total else 0.0

def sharp_social_score(team_name: str) -> float:
    """Estimate sharp sentiment from Reddit posts for the team."""
    posts = fetch_reddit_posts(team_name)
    return _call_sentiment_llm(posts, mode="sharp")

def hype_trend_score(team_name: str) -> float:
    """Estimate public hype/enthusiasm from Reddit posts."""
    posts = fetch_reddit_posts(team_name)
    return _call_sentiment_llm(posts, mode="hype")

def lineup_risk_score(team_name: str) -> float:
    """Estimate lineup/injury risk sentiment from Reddit posts."""
    posts = fetch_reddit_posts(team_name)
    return _call_sentiment_llm(posts, mode="risk")
