import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import GROUP_LETTERS, N_SIMULATIONS, PROCESSED_DIR, RANDOM_STATE
from src.helpers import logger


class KnockoutStageSimulator:
    def __init__(self, model, feature_cols: list[str], n_simulations: int = N_SIMULATIONS, imputer=None):
        self.model = model
        self.feature_cols = feature_cols
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.imputer = imputer
        if self.imputer is None:
            imputer_path = PROCESSED_DIR / "models" / "imputer.joblib"
            if imputer_path.exists():
                self.imputer = joblib.load(imputer_path)

    def predict_knockout_match(self, home: str, away: str, wc_features: pd.DataFrame) -> np.ndarray:
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
            col_mean = np.nanmean(X)
            X[nan_mask] = col_mean if not np.isnan(col_mean) else 0.0

        try:
            proba = self.model.predict_proba(X)[0]
            if len(proba) == 3:
                home_win_prob = proba[2]
                away_win_prob = proba[0]
                draw_prob = proba[1]
                probs = np.array([home_win_prob, draw_prob, away_win_prob])
                if swapped:
                    probs = np.array([probs[2], probs[1], probs[0]])
                probs = probs / probs.sum()
                return probs
        except Exception:
            pass

        return np.array([0.4, 0.3, 0.3])

    def simulate_knockout_match(self, home: str, away: str, wc_features: pd.DataFrame) -> str:
        probs = self.predict_knockout_match(home, away, wc_features)

        probs = probs / probs.sum()

        draw_prob = probs[1]
        extra_draw_reduction = draw_prob * 0.7
        home_win_prob = probs[0] + extra_draw_reduction * 0.55
        away_win_prob = probs[2] + extra_draw_reduction * 0.45

        total = home_win_prob + away_win_prob
        if total <= 0:
            return home
        home_win_prob /= total
        away_win_prob /= total

        home_win_prob = np.clip(home_win_prob, 0.001, 0.999)
        away_win_prob = 1.0 - home_win_prob

        p = np.array([home_win_prob, away_win_prob])
        p = p / p.sum()

        outcome = self.rng.choice(["home_win", "away_win"], p=p)

        return home if outcome == "home_win" else away

    def build_round_of_32_bracket(self, group_results: list[pd.DataFrame], third_place_teams: list[str]) -> dict:
        bracket = {}

        group_winners = {}
        group_runners_up = {}
        group_thirds = {}

        for group_df in group_results:
            if len(group_df) < 3:
                continue

            group_letter = group_df.iloc[0]["group"]
            group_winners[group_letter] = group_df.iloc[0]["team"]
            group_runners_up[group_letter] = group_df.iloc[1]["team"]
            if len(group_df) >= 3:
                group_thirds[group_letter] = group_df.iloc[2]["team"]

        third_qualified = {}
        for i, team in enumerate(third_place_teams):
            for g, t in group_thirds.items():
                if t == team:
                    third_qualified[i + 1] = (g, team)

        bracket_matches = {
            "R32_1": (group_winners.get("A", "?"), group_runners_up.get("B", "?")),
            "R32_2": (group_winners.get("B", "?"), group_runners_up.get("A", "?")),
            "R32_3": (group_winners.get("C", "?"), group_runners_up.get("D", "?")),
            "R32_4": (group_winners.get("D", "?"), group_runners_up.get("C", "?")),
            "R32_5": (group_winners.get("E", "?"), group_runners_up.get("F", "?")),
            "R32_6": (group_winners.get("F", "?"), group_runners_up.get("E", "?")),
            "R32_7": (group_winners.get("G", "?"), group_runners_up.get("H", "?")),
            "R32_8": (group_winners.get("H", "?"), group_runners_up.get("G", "?")),
            "R32_9": (group_winners.get("I", "?"), group_runners_up.get("J", "?")),
            "R32_10": (group_winners.get("J", "?"), group_runners_up.get("I", "?")),
            "R32_11": (group_winners.get("K", "?"), group_runners_up.get("L", "?")),
            "R32_12": (group_winners.get("L", "?"), group_runners_up.get("K", "?")),
        }

        if third_qualified:
            for i in range(min(len(third_qualified), 8)):
                match_num = 13 + i
                bracket_matches[f"R32_{match_num}"] = (
                    group_winners.get(GROUP_LETTERS[i] if i < len(GROUP_LETTERS) else "?", "?"),
                    third_qualified.get(i + 1, ("?", "?"))[1],
                )

        return bracket_matches

    def simulate_knockout_round(self, matches: dict, wc_features: pd.DataFrame) -> list[str]:
        winners = []
        for match_key, (home, away) in matches.items():
            if home == "?" or away == "?":
                winners.append(home if home != "?" else away)
                continue

            winner = self.simulate_knockout_match(home, away, wc_features)
            winners.append(winner)

        return winners

    def simulate_full_knockout(
        self,
        group_results: list[pd.DataFrame],
        third_place_teams: list[str],
        wc_features: pd.DataFrame,
    ) -> dict:
        bracket = self.build_round_of_32_bracket(group_results, third_place_teams)

        ro32_winners = self.simulate_knockout_round(bracket, wc_features)

        ro16_matches = {}
        for i in range(0, len(ro32_winners) - 1, 2):
            match_key = f"R16_{i // 2 + 1}"
            ro16_matches[match_key] = (ro32_winners[i], ro32_winners[i + 1])

        ro16_winners = self.simulate_knockout_round(ro16_matches, wc_features)

        qf_matches = {}
        for i in range(0, len(ro16_winners) - 1, 2):
            match_key = f"QF_{i // 2 + 1}"
            qf_matches[match_key] = (ro16_winners[i], ro16_winners[i + 1])

        qf_winners = self.simulate_knockout_round(qf_matches, wc_features)

        sf_matches = {
            "SF_1": (qf_winners[0], qf_winners[1]),
            "SF_2": (qf_winners[2], qf_winners[3]) if len(qf_winners) > 3 else (qf_winners[0], qf_winners[1]),
        }
        sf_winners = self.simulate_knockout_round(sf_matches, wc_features)

        final_match = {"Final": (sf_winners[0], sf_winners[1])}
        champion = self.simulate_knockout_round(final_match, wc_features)[0]

        return {
            "ro32": ro32_winners,
            "ro16": ro16_winners,
            "qf": qf_winners,
            "sf": sf_winners,
            "final": champion,
        }
