import pytest
from odd_utils import american_odds_to_prob, american_odds_to_payout


def test_american_odds_to_prob_positive():
    assert abs(american_odds_to_prob(150) - 0.4) < 1e-6
    assert abs(american_odds_to_prob(100) - 0.5) < 1e-6


def test_american_odds_to_prob_negative():
    assert abs(american_odds_to_prob(-200) - 0.666666) < 1e-5
    assert abs(american_odds_to_prob(-110) - 0.523809) < 1e-5


def test_american_odds_to_payout_positive():
    assert abs(american_odds_to_payout(150, 100) - 250) < 1e-6
    assert abs(american_odds_to_payout(200, 50) - 150) < 1e-6


def test_american_odds_to_payout_negative():
    assert abs(american_odds_to_payout(-200, 100) - 150) < 1e-6
    assert abs(american_odds_to_payout(-110, 22) - 42) < 1e-2


def test_american_odds_to_payout_default_stake():
    assert abs(american_odds_to_payout(100) - 2.0) < 1e-6
    assert abs(american_odds_to_payout(-200) - 1.5) < 1e-6
