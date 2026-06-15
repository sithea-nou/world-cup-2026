import pytest
import numpy as np
import pandas as pd

from src.config import GROUP_LETTERS, BEST_THIRD_ADVANCE, ADVANCE_PER_GROUP


@pytest.fixture
def sample_groups():
    teams_per_group = {
        "A": ["Team_A1", "Team_A2", "Team_A3", "Team_A4"],
        "B": ["Team_B1", "Team_B2", "Team_B3", "Team_B4"],
        "C": ["Team_C1", "Team_C2", "Team_C3", "Team_C4"],
    }
    records = []
    for group, teams in teams_per_group.items():
        for i, team in enumerate(teams):
            records.append({"group": group, "team": team, "pot": i + 1})
    return pd.DataFrame(records)


@pytest.fixture
def sample_group_results():
    results = []
    for group in ["A", "B", "C"]:
        for i, (team, points) in enumerate(
            [
                (f"Team_{group}1", 7),
                (f"Team_{group}2", 4),
                (f"Team_{group}3", 3),
                (f"Team_{group}4", 1),
            ]
        ):
            results.append(
                {
                    "team": team,
                    "group": group,
                    "points": points,
                    "goals_for": points * 2,
                    "goals_against": points,
                    "goal_diff": points,
                    "wins": points // 3,
                    "draws": points % 3,
                    "losses": 3 - points // 3 - (points % 3 if points % 3 < 3 else 0),
                    "position": i + 1,
                }
            )
    return pd.DataFrame(results)


class TestGroupStage:
    def test_group_letters(self):
        assert len(GROUP_LETTERS) == 12
        assert GROUP_LETTERS[0] == "A"
        assert GROUP_LETTERS[-1] == "L"

    def test_advance_per_group(self):
        assert ADVANCE_PER_GROUP == 2

    def test_best_third_advance(self):
        assert BEST_THIRD_ADVANCE == 8

    def test_sample_groups_structure(self, sample_groups):
        assert len(sample_groups) == 12
        assert "group" in sample_groups.columns
        assert "team" in sample_groups.columns

    def test_group_results_positions(self, sample_group_results):
        for group in sample_group_results["group"].unique():
            group_data = sample_group_results[sample_group_results["group"] == group]
            group_data = group_data.sort_values("position")
            positions = group_data["position"].tolist()
            assert positions == [1, 2, 3, 4]


class TestKnockoutStage:
    def test_knockout_match_returns_winner(self):
        from src.simulation.knockout_stage import KnockoutStageSimulator

        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.2, 0.2, 0.6]])

            def predict(self, X):
                return np.array([2])

        mock_model = MockModel()
        feature_cols = ["elo_home", "elo_away"]

        wc_features = pd.DataFrame({
            "home_team": ["Team1"],
            "away_team": ["Team2"],
            "elo_home": [1500],
            "elo_away": [1400],
        })

        simulator = KnockoutStageSimulator(mock_model, feature_cols, n_simulations=1)
        result = simulator.simulate_knockout_match("Team1", "Team2", wc_features)
        assert isinstance(result, str)
        assert result in ["Team1", "Team2"]


class TestSimulationProbabilities:
    def test_probability_monotonicity(self):
        results = pd.DataFrame(
            [
                {"team": "Best", "prob_ro32": 0.9, "prob_ro16": 0.7, "prob_qf": 0.5, "prob_sf": 0.3, "prob_final": 0.15, "prob_winner": 0.08},
                {"team": "Medium", "prob_ro32": 0.7, "prob_ro16": 0.5, "prob_qf": 0.3, "prob_sf": 0.15, "prob_final": 0.05, "prob_winner": 0.02},
                {"team": "Worst", "prob_ro32": 0.4, "prob_ro16": 0.2, "prob_qf": 0.1, "prob_sf": 0.03, "prob_final": 0.01, "prob_winner": 0.003},
            ]
        )

        for _, row in results.iterrows():
            assert row["prob_winner"] <= row["prob_final"]
            assert row["prob_final"] <= row["prob_sf"]
            assert row["prob_sf"] <= row["prob_qf"]
            assert row["prob_qf"] <= row["prob_ro16"]
            assert row["prob_ro16"] <= row["prob_ro32"]

    def test_all_probabilities_between_0_and_1(self):
        results = pd.DataFrame(
            [
                {"team": "Team1", "prob_ro32": 0.9, "prob_ro16": 0.7, "prob_qf": 0.5, "prob_sf": 0.3, "prob_final": 0.15, "prob_winner": 0.08},
            ]
        )

        prob_cols = ["prob_ro32", "prob_ro16", "prob_qf", "prob_sf", "prob_final", "prob_winner"]
        for col in prob_cols:
            assert (results[col] >= 0).all()
            assert (results[col] <= 1).all()