import pytest
import numpy as np
import pandas as pd

from src.features.elo import EloRatingSystem
from src.config import ELO_INITIAL_RATING, ELO_HOME_ADVANTAGE


@pytest.fixture
def sample_matches():
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "home_team": "Brazil", "away_team": "Argentina", "home_score": 2, "away_score": 1, "tournament": "Friendly", "neutral": False},
            {"date": "2024-02-01", "home_team": "Argentina", "away_team": "France", "home_score": 3, "away_score": 2, "tournament": "Friendly", "neutral": False},
            {"date": "2024-03-01", "home_team": "Germany", "away_team": "Brazil", "home_score": 1, "away_score": 1, "tournament": "Friendly", "neutral": True},
            {"date": "2024-04-01", "home_team": "France", "away_team": "Germany", "home_score": 0, "away_score": 2, "tournament": "Friendly", "neutral": False},
            {"date": "2024-05-01", "home_team": "Brazil", "away_team": "France", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": True},
        ]
    )


@pytest.fixture
def elo_system():
    return EloRatingSystem()


class TestEloInit:
    def test_initial_rating(self):
        elo = EloRatingSystem()
        assert elo.ratings == {}

    def test_custom_params(self):
        elo = EloRatingSystem(k_factor=60, home_advantage=80, initial_rating=1200)
        assert elo.k_factor == 60
        assert elo.home_advantage == 80
        assert elo.initial_rating == 1200


class TestEloComputation:
    def test_compute_elo_ratings(self, elo_system, sample_matches):
        sample_matches["date"] = pd.to_datetime(sample_matches["date"])
        ratings = elo_system.compute_elo_ratings(sample_matches)

        assert len(ratings) == 4
        assert "team" in ratings.columns
        assert "elo" in ratings.columns
        assert all(ratings["elo"] > 0)

    def test_elo_changes_after_matches(self, elo_system, sample_matches):
        sample_matches["date"] = pd.to_datetime(sample_matches["date"])
        elo_system.compute_elo_ratings(sample_matches)

        assert elo_system.ratings["Brazil"] != ELO_INITIAL_RATING
        assert elo_system.ratings["Argentina"] != ELO_INITIAL_RATING

    def test_get_team_rating(self, elo_system, sample_matches):
        sample_matches["date"] = pd.to_datetime(sample_matches["date"])
        elo_system.compute_elo_ratings(sample_matches)

        rating = elo_system.get_team_rating("Brazil")
        assert isinstance(rating, float)
        assert rating > 0

    def test_unknown_team_gets_initial(self, elo_system):
        rating = elo_system.get_team_rating("UnknownTeam")
        assert rating == ELO_INITIAL_RATING


class TestEloProbability:
    def test_probability_bounds(self, elo_system):
        probs = elo_system.predict_match_probability(1500, 1500, neutral=True)
        assert 0 < probs["home_win"] < 1
        assert 0 < probs["draw"] < 1
        assert 0 < probs["away_win"] < 1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_equal_ratings_fair(self, elo_system):
        probs = elo_system.predict_match_probability(1500, 1500, neutral=True)
        assert abs(probs["home_win"] - probs["away_win"]) < 0.01

    def test_higher_elo_more_likely(self, elo_system):
        probs_high = elo_system.predict_match_probability(1800, 1500, neutral=True)
        assert probs_high["home_win"] > probs_high["away_win"]

    def test_home_advantage_effect(self, elo_system):
        probs_neutral = elo_system.predict_match_probability(1500, 1500, neutral=True)
        probs_home = elo_system.predict_match_probability(1500, 1500, neutral=False)
        assert probs_home["home_win"] > probs_neutral["home_win"]

    def test_probability_sum(self, elo_system):
        for home_elo in [1200, 1500, 1800]:
            for away_elo in [1200, 1500, 1800]:
                probs = elo_system.predict_match_probability(home_elo, away_elo, neutral=True)
                assert abs(sum(probs.values()) - 1.0) < 0.02


class TestEloKFactor:
    def test_k_factor_by_tournament(self, elo_system):
        assert elo_system._get_k_factor("FIFA World Cup") == 80
        assert elo_system._get_k_factor("FIFA World Cup qualification") == 60
        assert elo_system._get_k_factor("Friendly") == 40
        assert elo_system._get_k_factor("Some Other Tournament") == 50


class TestEloTopRatings:
    def test_top_ratings(self, elo_system, sample_matches):
        sample_matches["date"] = pd.to_datetime(sample_matches["date"])
        elo_system.compute_elo_ratings(sample_matches)

        top = elo_system.get_top_ratings(3)
        assert len(top) == 3
        assert top.iloc[0]["elo"] >= top.iloc[1]["elo"]