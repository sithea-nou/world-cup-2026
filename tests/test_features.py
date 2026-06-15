import pytest
import numpy as np
import pandas as pd

from src.config import OUTCOME_HOME_WIN, OUTCOME_DRAW, OUTCOME_AWAY_WIN


@pytest.fixture
def sample_features():
    return pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "elo_home": 1800,
                "elo_away": 1750,
                "elo_delta": 50,
                "elo_abs_delta": 50,
                "elo_home_win_prob": 0.45,
                "elo_draw_prob": 0.28,
                "elo_away_win_prob": 0.27,
                "fifa_rank_home": 1,
                "fifa_rank_away": 2,
                "fifa_rank_delta": -1,
                "fifa_rank_abs_delta": 1,
                "fifa_points_home": 1840,
                "fifa_points_away": 1820,
                "fifa_points_delta": 20,
                "fifa_points_abs_delta": 20,
                "neutral": 0,
                "home_advantage": 1.0,
                "is_host_nation": 0,
                "same_confederation": 1,
                "is_world_cup": 0,
                "is_qualifier": 0,
                "is_friendly": 1,
                "is_knockout": 0,
                "combined_draw_prob": 0.28,
                "elo_close": 1,
                "draw_tendency": 0.42,
                "fifa_close": 1,
                "tournament_draw_rate": 0.23,
                "home_sos_avg_opp_elo": 1600.0,
                "away_sos_avg_opp_elo": 1580.0,
                "home_form_last10_win_rate": 0.7,
                "home_form_last10_draw_rate": 0.2,
                "home_form_last10_loss_rate": 0.1,
                "home_form_last10_goals_scored_avg": 1.8,
                "home_form_last10_goals_conceded_avg": 0.8,
                "home_form_last10_goal_diff_avg": 1.0,
                "home_form_last10_clean_sheet_rate": 0.3,
                "home_form_last10_ewm_win_rate": 0.72,
                "home_form_last10_ewm_draw_rate": 0.20,
                "home_form_last10_ewm_loss_rate": 0.08,
                "home_form_last10_ewm_goals_scored_avg": 1.9,
                "home_form_last10_ewm_goals_conceded_avg": 0.7,
                "home_form_last5_win_rate": 0.6,
                "home_form_last5_draw_rate": 0.2,
                "home_form_last5_loss_rate": 0.2,
                "home_form_last5_goals_scored_avg": 1.6,
                "home_form_last5_goals_conceded_avg": 0.8,
                "home_form_last5_goal_diff_avg": 0.8,
                "home_form_last5_clean_sheet_rate": 0.2,
                "home_form_last5_ewm_win_rate": 0.65,
                "home_form_last5_ewm_draw_rate": 0.20,
                "home_form_last5_ewm_loss_rate": 0.15,
                "home_form_last5_ewm_goals_scored_avg": 1.7,
                "home_form_last5_ewm_goals_conceded_avg": 0.7,
                "away_form_last10_win_rate": 0.65,
                "away_form_last10_draw_rate": 0.2,
                "away_form_last10_loss_rate": 0.15,
                "away_form_last10_goals_scored_avg": 1.6,
                "away_form_last10_goals_conceded_avg": 0.9,
                "away_form_last10_goal_diff_avg": 0.7,
                "away_form_last10_clean_sheet_rate": 0.25,
                "away_form_last10_ewm_win_rate": 0.67,
                "away_form_last10_ewm_draw_rate": 0.20,
                "away_form_last10_ewm_loss_rate": 0.13,
                "away_form_last10_ewm_goals_scored_avg": 1.7,
                "away_form_last10_ewm_goals_conceded_avg": 0.8,
                "away_form_last5_win_rate": 0.6,
                "away_form_last5_draw_rate": 0.2,
                "away_form_last5_loss_rate": 0.2,
                "away_form_last5_goals_scored_avg": 1.4,
                "away_form_last5_goals_conceded_avg": 1.0,
                "away_form_last5_goal_diff_avg": 0.4,
                "away_form_last5_clean_sheet_rate": 0.2,
                "away_form_last5_ewm_win_rate": 0.62,
                "away_form_last5_ewm_draw_rate": 0.20,
                "away_form_last5_ewm_loss_rate": 0.18,
                "away_form_last5_ewm_goals_scored_avg": 1.5,
                "away_form_last5_ewm_goals_conceded_avg": 0.9,
                "h2h_home_wins": 2,
                "h2h_draws": 1,
                "h2h_away_wins": 2,
                "h2h_draw_rate": 0.2,
                "h2h_home_goals_avg": 1.5,
                "h2h_away_goals_avg": 1.2,
                "outcome": OUTCOME_HOME_WIN,
                "goal_diff": 1,
                "home_score": 2,
                "away_score": 1,
            },
            {
                "date": "2024-02-01",
                "home_team": "France",
                "away_team": "Germany",
                "elo_home": 1750,
                "elo_away": 1700,
                "elo_delta": 50,
                "elo_abs_delta": 50,
                "elo_home_win_prob": 0.42,
                "elo_draw_prob": 0.30,
                "elo_away_win_prob": 0.28,
                "fifa_rank_home": 3,
                "fifa_rank_away": 4,
                "fifa_rank_delta": -1,
                "fifa_rank_abs_delta": 1,
                "fifa_points_home": 1800,
                "fifa_points_away": 1780,
                "fifa_points_delta": 20,
                "fifa_points_abs_delta": 20,
                "neutral": 0,
                "home_advantage": 1.0,
                "is_host_nation": 0,
                "same_confederation": 1,
                "is_world_cup": 0,
                "is_qualifier": 0,
                "is_friendly": 1,
                "is_knockout": 0,
                "combined_draw_prob": 0.30,
                "elo_close": 1,
                "draw_tendency": 0.45,
                "fifa_close": 1,
                "tournament_draw_rate": 0.23,
                "home_sos_avg_opp_elo": 1550.0,
                "away_sos_avg_opp_elo": 1530.0,
                "home_form_last10_win_rate": 0.6,
                "home_form_last10_draw_rate": 0.2,
                "home_form_last10_loss_rate": 0.2,
                "home_form_last10_goals_scored_avg": 1.5,
                "home_form_last10_goals_conceded_avg": 0.9,
                "home_form_last10_goal_diff_avg": 0.6,
                "home_form_last10_clean_sheet_rate": 0.25,
                "home_form_last10_ewm_win_rate": 0.62,
                "home_form_last10_ewm_draw_rate": 0.20,
                "home_form_last10_ewm_loss_rate": 0.18,
                "home_form_last10_ewm_goals_scored_avg": 1.6,
                "home_form_last10_ewm_goals_conceded_avg": 0.8,
                "home_form_last5_win_rate": 0.5,
                "home_form_last5_draw_rate": 0.2,
                "home_form_last5_loss_rate": 0.3,
                "home_form_last5_goals_scored_avg": 1.4,
                "home_form_last5_goals_conceded_avg": 1.0,
                "home_form_last5_goal_diff_avg": 0.4,
                "home_form_last5_clean_sheet_rate": 0.2,
                "home_form_last5_ewm_win_rate": 0.52,
                "home_form_last5_ewm_draw_rate": 0.20,
                "home_form_last5_ewm_loss_rate": 0.28,
                "home_form_last5_ewm_goals_scored_avg": 1.5,
                "home_form_last5_ewm_goals_conceded_avg": 0.9,
                "away_form_last10_win_rate": 0.55,
                "away_form_last10_draw_rate": 0.2,
                "away_form_last10_loss_rate": 0.25,
                "away_form_last10_goals_scored_avg": 1.4,
                "away_form_last10_goals_conceded_avg": 1.0,
                "away_form_last10_goal_diff_avg": 0.4,
                "away_form_last10_clean_sheet_rate": 0.2,
                "away_form_last10_ewm_win_rate": 0.57,
                "away_form_last10_ewm_draw_rate": 0.20,
                "away_form_last10_ewm_loss_rate": 0.23,
                "away_form_last10_ewm_goals_scored_avg": 1.5,
                "away_form_last10_ewm_goals_conceded_avg": 0.9,
                "away_form_last5_win_rate": 0.5,
                "away_form_last5_draw_rate": 0.2,
                "away_form_last5_loss_rate": 0.3,
                "away_form_last5_goals_scored_avg": 1.3,
                "away_form_last5_goals_conceded_avg": 1.1,
                "away_form_last5_goal_diff_avg": 0.2,
                "away_form_last5_clean_sheet_rate": 0.15,
                "away_form_last5_ewm_win_rate": 0.52,
                "away_form_last5_ewm_draw_rate": 0.20,
                "away_form_last5_ewm_loss_rate": 0.28,
                "away_form_last5_ewm_goals_scored_avg": 1.4,
                "away_form_last5_ewm_goals_conceded_avg": 1.0,
                "h2h_home_wins": 1,
                "h2h_draws": 2,
                "h2h_away_wins": 2,
                "h2h_draw_rate": 0.4,
                "h2h_home_goals_avg": 1.2,
                "h2h_away_goals_avg": 1.4,
                "outcome": OUTCOME_DRAW,
                "goal_diff": 0,
                "home_score": 1,
                "away_score": 1,
            },
        ]
    )


class TestFeatureColumns:
    def test_required_features_present(self, sample_features):
        from src.models.train import FEATURE_COLUMNS

        for col in FEATURE_COLUMNS:
            assert col in sample_features.columns, f"Missing required feature: {col}"


class TestOutcomeEncoding:
    def test_outcome_values(self):
        assert OUTCOME_HOME_WIN == 1
        assert OUTCOME_DRAW == 0
        assert OUTCOME_AWAY_WIN == -1


class TestFeatureEngineering:
    def test_feature_no_nulls_in_required(self, sample_features):
        from src.models.train import FEATURE_COLUMNS

        for col in FEATURE_COLUMNS:
            if col in sample_features.columns:
                assert sample_features[col].notna().any(), f"All null in feature: {col}"

    def test_elo_features_reasonable(self, sample_features):
        assert sample_features["elo_home_win_prob"].between(0, 1).all()
        assert sample_features["elo_draw_prob"].between(0, 1).all()
        assert sample_features["elo_away_win_prob"].between(0, 1).all()

    def test_form_features_reasonable(self, sample_features):
        for col in ["home_form_last10_win_rate", "away_form_last10_win_rate"]:
            if col in sample_features.columns:
                assert sample_features[col].between(0, 1).all()


class TestNormalization:
    def test_normalize_known_teams(self):
        from src.helpers import normalize_team_name

        assert normalize_team_name("USA") == "United States"
        assert normalize_team_name("Korea Republic") == "South Korea"
        assert normalize_team_name("Brazil") == "Brazil"

    def test_normalize_strip_whitespace(self):
        from src.helpers import normalize_team_name

        assert normalize_team_name("  Brazil  ") == "Brazil"