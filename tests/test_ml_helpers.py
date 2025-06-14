import ml
import requests
from datetime import datetime


def test_to_fixed_utc():
    dt = datetime(2024, 1, 2, 5, 30)
    assert ml.to_fixed_utc(dt) == "2024-01-02T12:00:00Z"


def test_fetch_historical_h2h_odds(monkeypatch):
    called = {}

    class Resp:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"id": 1}]

    def fake_get(url, params=None, timeout=30):
        called["url"] = url
        called["params"] = params
        return Resp()

    monkeypatch.setenv("THE_ODDS_API_KEY", "key")
    monkeypatch.setattr(requests, "get", fake_get)

    data = ml.fetch_historical_h2h_odds("baseball_mlb", "2024-01-02T12:00:00Z")
    assert data == [{"id": 1}]
    assert "baseball_mlb" in called["url"]
    assert called["params"]["markets"] == "h2h"

