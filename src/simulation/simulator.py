import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    GROUP_LETTERS,
    N_SIMULATIONS,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_STATE,
    WC_GROUP_DRAW_RATE,
)
from src.helpers import logger
from src.simulation.group_stage import GroupStageSimulator
from src.simulation.knockout_stage import KnockoutStageSimulator


DEFAULT_SIMULATION_MODEL = "xgboost"


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


class WorldCupSimulator:
    def __init__(
        self,
        model=None,
        feature_cols: list[str] | None = None,
        n_simulations: int = N_SIMULATIONS,
        model_name: str = DEFAULT_SIMULATION_MODEL,
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.n_simulations = n_simulations
        self.model_name = model_name
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.imputer = None
        imputer_path = PROCESSED_DIR / "models" / "imputer.joblib"
        if imputer_path.exists():
            self.imputer = joblib.load(imputer_path)

        self._load_data()
        self._load_model()

    def _load_data(self):
        groups_path = RAW_DIR / "wc2026_groups.csv"
        if groups_path.exists():
            self.groups_df = pd.read_csv(groups_path)
            logger.info(f"Loaded {len(self.groups_df)} group entries from {groups_path}")
        else:
            logger.warning(f"Groups file not found at {groups_path}")
            self.groups_df = pd.DataFrame()

        fixtures_path = RAW_DIR / "wc2026_fixtures.csv"
        if fixtures_path.exists():
            self.fixtures_df = pd.read_csv(fixtures_path)
            logger.info(f"Loaded {len(self.fixtures_df)} fixtures from {fixtures_path}")
        else:
            logger.warning(f"Fixtures file not found at {fixtures_path}")
            self.fixtures_df = pd.DataFrame()

        features_path = PROCESSED_DIR / "wc2026_match_features.parquet"
        if features_path.exists():
            self.wc_features = pd.read_parquet(features_path)
            logger.info(f"Loaded {len(self.wc_features)} WC 2026 features from {features_path}")
        else:
            logger.warning(f"WC 2026 features not found at {features_path}")
            self.wc_features = pd.DataFrame()

        live_path = RAW_DIR / "wc2026_results_live.csv"
        if live_path.exists():
            self.live_results = pd.read_csv(live_path)
            self.live_results = self.live_results[
                self.live_results["home_score"].notna() & self.live_results["away_score"].notna()
            ]
            logger.info(f"Loaded {len(self.live_results)} live results")
        else:
            self.live_results = pd.DataFrame()

    def _load_model(self):
        if self.model is not None and self.feature_cols is not None:
            return

        models_dir = PROCESSED_DIR / "models"
        feature_cols_path = models_dir / "feature_columns.joblib"

        if not feature_cols_path.exists():
            logger.error("No feature columns found. Run training pipeline first.")
            return

        self.feature_cols = joblib.load(feature_cols_path)

        # Prefer the specified fast model (e.g., XGBoost) for simulation speed.
        # Fall back to the calibrated ensemble if not available.
        candidate_paths = [
            models_dir / f"{self.model_name}.joblib",
            models_dir / "best_model.joblib",
            models_dir / "xgboost.joblib",
        ]
        chosen_path = None
        for p in candidate_paths:
            if p.exists():
                chosen_path = p
                break

        if chosen_path is None:
            logger.error("No model found. Run training pipeline first.")
            return

        self.model = joblib.load(chosen_path)
        logger.info(f"Loaded simulation model from {chosen_path}")

    def _apply_live_results(self, group_results: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
        if self.live_results.empty:
            return group_results, []

        for i, group_df in enumerate(group_results):
            for _, match in self.live_results.iterrows():
                home = match["home_team"]
                away = match["away_team"]
                home_score = int(match["home_score"])
                away_score = int(match["away_score"])

                group_teams = group_df["team"].tolist()
                if home in group_teams and away in group_teams:
                    team_row_home = group_df[group_df["team"] == home].index[0]
                    team_row_away = group_df[group_df["team"] == away].index[0]

                    if home_score > away_score:
                        group_df.at[team_row_home, "points"] = group_df.at[team_row_home, "points"] + 3
                        group_df.at[team_row_home, "wins"] = group_df.at[team_row_home, "wins"] + 1
                        group_df.at[team_row_away, "losses"] = group_df.at[team_row_away, "losses"] + 1
                    elif home_score < away_score:
                        group_df.at[team_row_away, "points"] = group_df.at[team_row_away, "points"] + 3
                        group_df.at[team_row_away, "wins"] = group_df.at[team_row_away, "wins"] + 1
                        group_df.at[team_row_home, "losses"] = group_df.at[team_row_home, "losses"] + 1
                    else:
                        group_df.at[team_row_home, "points"] = group_df.at[team_row_home, "points"] + 1
                        group_df.at[team_row_away, "points"] = group_df.at[team_row_away, "points"] + 1
                        group_df.at[team_row_home, "draws"] = group_df.at[team_row_home, "draws"] + 1
                        group_df.at[team_row_away, "draws"] = group_df.at[team_row_away, "draws"] + 1

                    group_df.at[team_row_home, "goals_for"] = group_df.at[team_row_home, "goals_for"] + home_score
                    group_df.at[team_row_home, "goals_against"] = group_df.at[team_row_home, "goals_against"] + away_score
                    group_df.at[team_row_away, "goals_for"] = group_df.at[team_row_away, "goals_for"] + away_score
                    group_df.at[team_row_away, "goals_against"] = group_df.at[team_row_away, "goals_against"] + home_score
                    group_df.at[team_row_home, "goal_diff"] = group_df.at[team_row_home, "goals_for"] - group_df.at[team_row_home, "goals_against"]
                    group_df.at[team_row_away, "goal_diff"] = group_df.at[team_row_away, "goals_for"] - group_df.at[team_row_away, "goals_against"]

            group_df = group_df.sort_values(
                ["points", "goal_diff", "goals_for"], ascending=False
            ).reset_index(drop=True)
            group_df["position"] = range(1, len(group_df) + 1)
            group_results[i] = group_df

        from src.simulation.group_stage import GroupStageSimulator
        temp_sim = GroupStageSimulator(self.model, self.feature_cols, self.groups_df)
        third_place_teams = temp_sim.determine_third_place_qualifiers(group_results)

        return group_results, third_place_teams

    def run_full_simulation(self) -> pd.DataFrame:
        logger.info(f"Running full tournament simulation ({self.n_simulations} iterations)...")

        if self.groups_df.empty:
            logger.error("No group data available")
            return pd.DataFrame()

        group_simulator = GroupStageSimulator(
            self.model, self.feature_cols, self.groups_df, self.n_simulations
        )
        knockout_simulator = KnockoutStageSimulator(
            self.model, self.feature_cols, self.n_simulations
        )

        all_teams = self.groups_df["team"].unique()
        round_counts = {team: {"ro32": 0, "ro16": 0, "qf": 0, "sf": 0, "final": 0, "winner": 0} for team in all_teams}

        group_advancement_counts = {}
        for group_letter in GROUP_LETTERS:
            group_teams = self.groups_df[self.groups_df["group"] == group_letter]["team"].tolist()
            for team in group_teams:
                group_advancement_counts[team] = {
                    "1st": 0, "2nd": 0, "3rd": 0, "4th": 0, "advance": 0, "group": group_letter,
                }

        for sim in tqdm(range(self.n_simulations), desc="Tournament simulations"):
            group_results, third_place_teams = group_simulator.simulate_all_groups(self.wc_features)

            if not self.live_results.empty:
                group_results, third_place_teams = self._apply_live_results(group_results)

            for group_df in group_results:
                for _, row in group_df.iterrows():
                    team = row["team"]
                    pos = row["position"]
                    if team in group_advancement_counts:
                        pos_key = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(pos, "4th")
                        group_advancement_counts[team][pos_key] += 1
                        if pos <= 2:
                            group_advancement_counts[team]["advance"] += 1
                        elif pos == 3 and team in third_place_teams:
                            group_advancement_counts[team]["advance"] += 1

            knockout_results = knockout_simulator.simulate_full_knockout(
                group_results, third_place_teams, self.wc_features
            )

            for team in knockout_results.get("ro32", []):
                if team in round_counts:
                    round_counts[team]["ro32"] += 1

            for team in knockout_results.get("ro16", []):
                if team in round_counts:
                    round_counts[team]["ro16"] += 1

            for team in knockout_results.get("qf", []):
                if team in round_counts:
                    round_counts[team]["qf"] += 1

            for team in knockout_results.get("sf", []):
                if team in round_counts:
                    round_counts[team]["sf"] += 1

            if "final" in knockout_results and isinstance(knockout_results["final"], str):
                finalist = knockout_results["final"]
                if finalist in round_counts:
                    round_counts[finalist]["final"] += 1

            champion = knockout_results.get("final", "")
            if champion in round_counts:
                round_counts[champion]["winner"] += 1

        results = []
        for team, counts in round_counts.items():
            results.append(
                {
                    "team": team,
                    "prob_ro32": counts["ro32"] / self.n_simulations,
                    "prob_ro16": counts["ro16"] / self.n_simulations,
                    "prob_qf": counts["qf"] / self.n_simulations,
                    "prob_sf": counts["sf"] / self.n_simulations,
                    "prob_final": counts["final"] / self.n_simulations,
                    "prob_winner": counts["winner"] / self.n_simulations,
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values("prob_winner", ascending=False).reset_index(drop=True)

        out_path = PROCESSED_DIR / "tournament_probabilities.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Tournament probabilities saved to {out_path}")

        group_probs = []
        for team, counts in group_advancement_counts.items():
            group_probs.append(
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

        group_df = pd.DataFrame(group_probs)
        group_df = group_df.sort_values(["group", "prob_advance"], ascending=[True, False]).reset_index(drop=True)

        group_out_path = PROCESSED_DIR / "group_stage_probabilities.csv"
        group_df.to_csv(group_out_path, index=False)
        logger.info(f"Group stage probabilities saved to {group_out_path}")

        return df

    def get_results(self) -> pd.DataFrame:
        results_path = PROCESSED_DIR / "tournament_probabilities.csv"
        if results_path.exists():
            return pd.read_csv(results_path)
        return pd.DataFrame()

    def save_results(self, results: pd.DataFrame = None):
        if results is None:
            results = self.get_results()
        if results.empty:
            logger.warning("No results to save")
            return

        out_path = PROCESSED_DIR / "tournament_probabilities.csv"
        results.to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")

    def _predict_match_proba(self, home: str, away: str) -> np.ndarray:
        feature_row = self.wc_features[
            (self.wc_features["home_team"] == home) & (self.wc_features["away_team"] == away)
        ]
        swapped = False

        if feature_row.empty:
            feature_row = self.wc_features[
                (self.wc_features["home_team"] == away) & (self.wc_features["away_team"] == home)
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
                probs = _calibrate_draw(probs, WC_GROUP_DRAW_RATE)
                return probs
        except Exception:
            pass

        return np.array([0.4, 0.3, 0.3])

    def predict_all_matches(self) -> pd.DataFrame:
        logger.info("Predicting all WC 2026 matches...")

        if self.fixtures_df.empty:
            logger.error("No fixtures available")
            return pd.DataFrame()

        predictions = []
        for _, fixture in self.fixtures_df.iterrows():
            home = fixture["home_team"]
            away = fixture["away_team"]
            group = fixture.get("group", "")

            probs = self._predict_match_proba(home, away)

            if probs[0] >= probs[1] and probs[0] >= probs[2]:
                prediction = "home_win"
            elif probs[2] >= probs[0] and probs[2] >= probs[1]:
                prediction = "away_win"
            else:
                prediction = "draw"

            predictions.append(
                {
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "prob_home_win": round(probs[0], 4),
                    "prob_draw": round(probs[1], 4),
                    "prob_away_win": round(probs[2], 4),
                    "prediction": prediction,
                }
            )

        df = pd.DataFrame(predictions)

        out_path = PROCESSED_DIR / "match_predictions.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Match predictions saved to {out_path} ({len(df)} matches)")

        return df


def run_simulation(
    n_simulations: int | None = None,
    model=None,
    feature_cols: list[str] | None = None,
    model_name: str = DEFAULT_SIMULATION_MODEL,
) -> pd.DataFrame:
    if n_simulations is None:
        n_simulations = N_SIMULATIONS

    simulator = WorldCupSimulator(
        n_simulations=n_simulations,
        model=model,
        feature_cols=feature_cols,
        model_name=model_name,
    )
    results = simulator.run_full_simulation()
    return results


if __name__ == "__main__":
    results = run_simulation()

    print("\nTournament Probabilities:")
    print(results.to_string(index=False))

    print(f"\nTop 10 winners:")
    print(results.head(10)[["team", "prob_winner"]].to_string(index=False))