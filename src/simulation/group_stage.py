import joblib
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm

from src.config import (
    ADVANCE_PER_GROUP,
    BEST_THIRD_ADVANCE,
    GROUP_LETTERS,
    HOST_NATIONS,
    N_SIMULATIONS,
    PROCESSED_DIR,
    RANDOM_STATE,
    WC_GROUP_DRAW_RATE,
)
from src.helpers import logger


class GroupStageSimulator:
    def __init__(self, model, feature_cols: list[str], groups_df: pd.DataFrame, n_simulations: int = N_SIMULATIONS, imputer=None):
        self.model = model
        self.feature_cols = feature_cols
        self.groups_df = groups_df
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.imputer = imputer
        if self.imputer is None:
            imputer_path = PROCESSED_DIR / "models" / "imputer.joblib"
            if imputer_path.exists():
                self.imputer = joblib.load(imputer_path)

    @staticmethod
    def _calibrate_draw(probs: np.ndarray, target_draw_rate: float) -> np.ndarray:
        draw_prob = probs[1]
        if draw_prob >= target_draw_rate:
            return probs
        boost = target_draw_rate - draw_prob
        new_draw = probs[1] + boost
        remaining = 1.0 - new_draw
        total_non_draw = probs[0] + probs[2]
        if total_non_draw <= 0:
            return probs
        scale = remaining / total_non_draw
        new_home = probs[0] * scale
        new_away = probs[2] * scale
        calibrated = np.array([new_home, new_draw, new_away])
        calibrated = calibrated / calibrated.sum()
        return calibrated

    def _predict_match(self, home: str, away: str, wc_features: pd.DataFrame) -> np.ndarray:
        feature_row = wc_features[
            (wc_features["home_team"] == home) & (wc_features["away_team"] == away)
        ]
        swapped = False

        if feature_row.empty:
            feature_row = wc_features[
                (wc_features["home_team"] == away) & (wc_features["away_team"] == home)
            ]
            swapped = True

        if feature_row.empty:
            return np.array([0.4, 0.3, 0.3])

        X = feature_row[self.feature_cols].values.flatten().reshape(1, -1)

        nan_mask = np.isnan(X)
        if nan_mask.any() and self.imputer is not None:
            X = self.imputer.transform(X)
        elif nan_mask.any():
            col_means = np.nanmean(X)
            X[nan_mask] = col_means if not np.isnan(col_means) else 0.0

        try:
            proba = self.model.predict_proba(X)[0]
            if len(proba) == 3:
                probs = np.array([proba[2], proba[1], proba[0]])
                if swapped:
                    probs = np.array([probs[2], probs[1], probs[0]])
                probs = probs / probs.sum()
                probs = self._calibrate_draw(probs, WC_GROUP_DRAW_RATE)
                return probs
        except Exception:
            pass

        return np.array([0.4, 0.3, 0.3])

    def simulate_match(self, home: str, away: str, wc_features: pd.DataFrame) -> tuple[str, int, int]:
        probs = self._predict_match(home, away, wc_features)

        probs = probs / probs.sum()

        probs = probs / probs.sum()
        outcome = self.rng.choice(["home_win", "draw", "away_win"], p=probs)

        if outcome == "home_win":
            home_goals = max(1, int(self.rng.poisson(1.5)))
            away_goals = max(0, home_goals - 1 - int(self.rng.integers(0, 2)))
            away_goals = max(0, away_goals)
            return outcome, home_goals, away_goals
        elif outcome == "away_win":
            away_goals = max(1, int(self.rng.poisson(1.3)))
            home_goals = max(0, away_goals - 1 - int(self.rng.integers(0, 2)))
            return outcome, home_goals, away_goals
        else:
            home_goals = int(self.rng.poisson(1.0))
            away_goals = home_goals
            return outcome, home_goals, away_goals

    def simulate_group(self, group_letter: str, wc_features: pd.DataFrame) -> pd.DataFrame:
        group_teams = self.groups_df[self.groups_df["group"] == group_letter]["team"].tolist()

        if len(group_teams) < 4:
            logger.warning(f"Group {group_letter} has fewer than 4 teams: {group_teams}")
            return pd.DataFrame()

        standings = {team: {"points": 0, "goals_for": 0, "goals_against": 0, "wins": 0, "draws": 0, "losses": 0} for team in group_teams}

        for home, away in combinations(group_teams, 2):
            outcome, home_goals, away_goals = self.simulate_match(home, away, wc_features)

            standings[home]["goals_for"] += home_goals
            standings[home]["goals_against"] += away_goals
            standings[away]["goals_for"] += away_goals
            standings[away]["goals_against"] += home_goals

            if outcome == "home_win":
                standings[home]["points"] += 3
                standings[home]["wins"] += 1
                standings[away]["losses"] += 1
            elif outcome == "away_win":
                standings[away]["points"] += 3
                standings[away]["wins"] += 1
                standings[home]["losses"] += 1
            else:
                standings[home]["points"] += 1
                standings[away]["points"] += 1
                standings[home]["draws"] += 1
                standings[away]["draws"] += 1

        results = []
        for team, stats in standings.items():
            results.append(
                {
                    "team": team,
                    "group": group_letter,
                    "points": stats["points"],
                    "goals_for": stats["goals_for"],
                    "goals_against": stats["goals_against"],
                    "goal_diff": stats["goals_for"] - stats["goals_against"],
                    "wins": stats["wins"],
                    "draws": stats["draws"],
                    "losses": stats["losses"],
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values(["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
        df["position"] = range(1, len(df) + 1)

        return df

    def determine_third_place_qualifiers(self, all_group_results: list[pd.DataFrame]) -> list[str]:
        third_place_teams = []
        for group_df in all_group_results:
            if len(group_df) >= 3:
                third = group_df.iloc[2]
                third_place_teams.append(
                    {
                        "team": third["team"],
                        "group": third["group"],
                        "points": third["points"],
                        "goal_diff": third["goal_diff"],
                        "goals_for": third["goals_for"],
                    }
                )

        third_df = pd.DataFrame(third_place_teams)
        third_df = third_df.sort_values(
            ["points", "goal_diff", "goals_for"], ascending=False
        ).reset_index(drop=True)

        return third_df.head(BEST_THIRD_ADVANCE)["team"].tolist()

    def simulate_all_groups(self, wc_features: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
        all_results = []
        for group_letter in GROUP_LETTERS:
            group_result = self.simulate_group(group_letter, wc_features)
            all_results.append(group_result)

        third_place_teams = self.determine_third_place_qualifiers(all_results)

        return all_results, third_place_teams

    def get_advancement_probabilities(self, wc_features: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Computing advancement probabilities over {self.n_simulations} simulations...")

        advancement_counts = {}

        for group_letter in GROUP_LETTERS:
            group_teams = self.groups_df[self.groups_df["group"] == group_letter]["team"].tolist()
            for team in group_teams:
                advancement_counts[team] = {
                    "1st": 0,
                    "2nd": 0,
                    "3rd": 0,
                    "4th": 0,
                    "advance": 0,
                    "group": group_letter,
                }

        for sim in tqdm(range(self.n_simulations), desc="Group simulations"):
            all_results = []
            for group_letter in GROUP_LETTERS:
                group_result = self.simulate_group(group_letter, wc_features)
                all_results.append(group_result)

            third_place_teams = self.determine_third_place_qualifiers(all_results)

            for group_df in all_results:
                for _, row in group_df.iterrows():
                    team = row["team"]
                    pos = row["position"]

                    if pos == 1:
                        advancement_counts[team]["1st"] += 1
                        advancement_counts[team]["advance"] += 1
                    elif pos == 2:
                        advancement_counts[team]["2nd"] += 1
                        advancement_counts[team]["advance"] += 1
                    elif pos == 3:
                        advancement_counts[team]["3rd"] += 1
                        if team in third_place_teams:
                            advancement_counts[team]["advance"] += 1
                    else:
                        advancement_counts[team]["4th"] += 1

        results = []
        for team, counts in advancement_counts.items():
            results.append(
                {
                    "team": team,
                    "group": counts["group"],
                    "prob_1st": counts["1st"] / self.n_simulations,
                    "prob_2nd": counts["2nd"] / self.n_simulations,
                    "prob_3rd": counts["3rd"] / self.n_simulations,
                    "prob_4th": counts["4th"] / self.n_simulations,
                    "prob_advance": counts["advance"] / self.n_simulations,
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values(["group", "prob_advance"], ascending=[True, False]).reset_index(drop=True)

        out_path = PROCESSED_DIR / "group_stage_probabilities.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Group stage probabilities saved to {out_path}")

        return df