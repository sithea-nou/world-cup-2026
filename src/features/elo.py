import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    ELO_DRAW_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL_RATING,
    K_FACTORS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
)
from src.helpers import logger, normalize_team_name


class EloRatingSystem:
    def __init__(
        self,
        k_factor: int = 40,
        home_advantage: int = ELO_HOME_ADVANTAGE,
        initial_rating: float = ELO_INITIAL_RATING,
        draw_factor: float = ELO_DRAW_FACTOR,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.draw_factor = draw_factor
        self.ratings: dict[str, float] = {}
        self.rating_history: list[dict] = []

    def _get_k_factor(self, tournament: str) -> int:
        tournament_lower = tournament.lower()
        best_match = None
        best_len = 0
        for key, val in K_FACTORS.items():
            if key.lower() in tournament_lower and len(key) > best_len:
                best_match = val
                best_len = len(key)
        return best_match if best_match is not None else K_FACTORS["default"]

    def _get_team_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def compute_elo_ratings(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Computing Elo ratings from {len(matches_df)} matches...")

        matches = matches_df.sort_values("date").reset_index(drop=True)

        for _, row in matches.iterrows():
            home = normalize_team_name(str(row.get("home_team", "")))
            away = normalize_team_name(str(row.get("away_team", "")))

            if home not in self.ratings:
                self.ratings[home] = self.initial_rating
            if away not in self.ratings:
                self.ratings[away] = self.initial_rating

            home_elo = self.ratings[home]
            away_elo = self.ratings[away]

            neutral = row.get("neutral", True)
            if isinstance(neutral, str):
                neutral = neutral.lower() == "true"
            neutral = bool(neutral)

            home_elo_adj = home_elo + (0 if neutral else self.home_advantage)

            tournament = str(row.get("tournament", "Friendly"))
            k = self._get_k_factor(tournament)

            prob_home = self._expected_score(home_elo_adj, away_elo)

            home_score = row.get("home_score", 0)
            away_score = row.get("away_score", 0)

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            if home_score > away_score:
                actual_home = 1.0
            elif home_score < away_score:
                actual_home = 0.0
            else:
                actual_home = 0.5

            self.ratings[home] = home_elo + k * (actual_home - prob_home)
            self.ratings[away] = away_elo + k * ((1 - actual_home) - (1 - prob_home))

            self.rating_history.append(
                {
                    "date": row.get("date"),
                    "team": home,
                    "elo": self.ratings[home],
                    "opponent": away,
                    "result": "win" if home_score > away_score else ("loss" if home_score < away_score else "draw"),
                }
            )
            self.rating_history.append(
                {
                    "date": row.get("date"),
                    "team": away,
                    "elo": self.ratings[away],
                    "opponent": home,
                    "result": "win" if away_score > home_score else ("loss" if away_score < home_score else "draw"),
                }
            )

        ratings_df = pd.DataFrame(
            [{"team": team, "elo": rating} for team, rating in sorted(self.ratings.items(), key=lambda x: -x[1])]
        )

        out_path = PROCESSED_DIR / "elo_ratings_current.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ratings_df.to_parquet(out_path, index=False)
        logger.info(f"Saved Elo ratings for {len(ratings_df)} teams to {out_path}")

        history_path = PROCESSED_DIR / "elo_ratings_history.parquet"
        history_df = pd.DataFrame(self.rating_history)
        history_df.to_parquet(history_path, index=False)
        logger.info(f"Saved {len(history_df)} rating history entries to {history_path}")

        return ratings_df

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def predict_match_probability(
        self, home_elo: float, away_elo: float, neutral: bool = True
    ) -> dict[str, float]:
        home_elo_adj = home_elo + (0 if neutral else self.home_advantage)

        expected_home = self._expected_score(home_elo_adj, away_elo)
        expected_away = 1 - expected_home

        raw_draw_prob = self.draw_factor * (1 - abs(expected_home - expected_away))
        draw_prob = min(raw_draw_prob, max(0.05, min(expected_home, expected_away)))
        draw_prob = min(draw_prob, 0.35)

        remaining = 1 - draw_prob
        home_prob = remaining * expected_home
        away_prob = remaining * expected_away

        total = home_prob + draw_prob + away_prob
        return {
            "home_win": home_prob / total,
            "draw": draw_prob / total,
            "away_win": away_prob / total,
        }

    def get_team_rating(self, team: str) -> float:
        team = normalize_team_name(team)
        return self._get_team_rating(team)

    def get_top_ratings(self, n: int = 30) -> pd.DataFrame:
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: -x[1])[:n]
        return pd.DataFrame(sorted_ratings, columns=["team", "elo"])


def validate_elo_against_results(ratings_df: pd.DataFrame, results_df: pd.DataFrame) -> dict:
    logger.info("Validating Elo ratings against match results...")

    results = results_df.copy()
    results = results.sort_values("date").reset_index(drop=True)

    correct = 0
    total = 0
    log_loss_sum = 0

    elo_system = EloRatingSystem()

    for _, row in results.iterrows():
        home = normalize_team_name(str(row.get("home_team", "")))
        away = normalize_team_name(str(row.get("away_team", "")))

        home_elo = elo_system.get_team_rating(home)
        away_elo = elo_system.get_team_rating(away)

        neutral = row.get("neutral", True)
        if isinstance(neutral, str):
            neutral = neutral.lower() == "true"

        probs = elo_system.predict_match_probability(home_elo, away_elo, neutral=bool(neutral))

        home_score = row.get("home_score", 0)
        away_score = row.get("away_score", 0)

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        predicted = max(probs, key=probs.get)
        if home_score > away_score:
            actual = "home_win"
        elif home_score < away_score:
            actual = "away_win"
        else:
            actual = "draw"

        if predicted == actual:
            correct += 1
        total += 1

        actual_prob = probs[actual]
        log_loss_sum -= np.log(max(actual_prob, 1e-15))

        elo_system.ratings[home] = home_elo
        elo_system.ratings[away] = away_elo

    accuracy = correct / total if total > 0 else 0
    avg_log_loss = log_loss_sum / total if total > 0 else float("inf")

    logger.info(f"Elo validation: accuracy={accuracy:.4f}, log_loss={avg_log_loss:.4f} ({total} matches)")

    return {"accuracy": accuracy, "log_loss": avg_log_loss, "total_matches": total}


if __name__ == "__main__":
    from src.scraping.download_kaggle import download_match_results

    matches_dir = download_match_results()
    matches_path = matches_dir / "results.csv"

    if matches_path.exists():
        matches_df = pd.read_csv(matches_path)
        matches_df["date"] = pd.to_datetime(matches_df["date"])

        elo_system = EloRatingSystem()
        ratings = elo_system.compute_elo_ratings(matches_df)

        print("\nTop 30 Elo Ratings:")
        print(ratings.head(30).to_string())

        print(f"\nMatch probabilities example (Argentina vs Brazil, neutral):")
        arg_elo = elo_system.get_team_rating("Argentina")
        bra_elo = elo_system.get_team_rating("Brazil")
        probs = elo_system.predict_match_probability(arg_elo, bra_elo, neutral=True)
        print(f"  Argentina win: {probs['home_win']:.3f}")
        print(f"  Draw: {probs['draw']:.3f}")
        print(f"  Brazil win: {probs['away_win']:.3f}")