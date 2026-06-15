import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.config import PROCESSED_DIR, RAW_DIR
from src.helpers import logger

warnings.filterwarnings("ignore", category=UserWarning)

LABEL_MAP = {1: 2, 0: 1, -1: 0}
LABEL_NAMES = ["away_win", "draw", "home_win"]


def validate_against_live(model, feature_cols: list[str]) -> dict:
    logger.info("Validating model against live WC 2026 results...")

    live_path = RAW_DIR / "wc2026_results_live.csv"
    if not live_path.exists():
        logger.warning("No live results found. Run scrape_live_results.py first.")
        return {}

    live_df = pd.read_csv(live_path)
    played = live_df[live_df["home_score"].notna() & live_df["away_score"].notna()].copy()

    if played.empty:
        logger.warning("No played matches with scores found in live results.")
        return {}

    logger.info(f"Found {len(played)} played WC 2026 matches to validate against")

    wc_features_path = PROCESSED_DIR / "wc2026_match_features.parquet"
    if wc_features_path.exists():
        wc_features = pd.read_parquet(wc_features_path)
    else:
        logger.warning("WC 2026 features not found. Run build_2026_features.py first.")
        return {}

    imputer = None
    imputer_path = PROCESSED_DIR / "models" / "imputer.joblib"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)

    available_cols = [c for c in feature_cols if c in wc_features.columns]
    missing_cols = [c for c in feature_cols if c not in wc_features.columns]

    if missing_cols:
        for col in missing_cols:
            wc_features[col] = 0.0

    validation_results = []
    correct = 0
    total = 0
    total_log_loss = 0

    for _, match in played.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        if home_score > away_score:
            actual = 2
        elif home_score < away_score:
            actual = 0
        else:
            actual = 1

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
            logger.debug(f"  No features for {home} vs {away}")
            continue

        X = feature_row[feature_cols].values.flatten().reshape(1, -1)

        if np.isnan(X).any() and imputer is not None:
            X = imputer.transform(X)
        elif np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, nan_mask.nonzero()[1])

        proba = model.predict_proba(X)[0]
        pred_class = model.predict(X)[0]

        if swapped:
            proba_swapped = np.array([proba[2], proba[1], proba[0]])
            proba = proba_swapped
            pred_class = {0: 2, 1: 1, 2: 0}[pred_class]

        is_correct = pred_class == actual
        if is_correct:
            correct += 1
        total += 1

        actual_prob = proba[actual]
        total_log_loss += -np.log(max(actual_prob, 1e-15))

        validation_results.append(
            {
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "actual_outcome": LABEL_NAMES[actual],
                "predicted_outcome": LABEL_NAMES[pred_class],
                "prob_home_win": proba[2],
                "prob_draw": proba[1],
                "prob_away_win": proba[0],
                "correct": is_correct,
            }
        )

    if total == 0:
        logger.warning("No matches could be validated")
        return {}

    accuracy = correct / total
    avg_log_loss = total_log_loss / total

    logger.info(f"Live validation results:")
    logger.info(f"  Matches: {total}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Log Loss: {avg_log_loss:.4f}")

    results_df = pd.DataFrame(validation_results)
    out_path = PROCESSED_DIR / "live_validation_report.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"  Saved to {out_path}")

    for _, r in results_df.iterrows():
        marker = "✓" if r["correct"] else "✗"
        logger.info(
            f"  {marker} {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']} "
            f"| Pred: {r['predicted_outcome']} | Actual: {r['actual_outcome']} "
            f"| H:{r['prob_home_win']:.2f} D:{r['prob_draw']:.2f} A:{r['prob_away_win']:.2f}"
        )

    return {
        "accuracy": accuracy,
        "log_loss": avg_log_loss,
        "total_matches": total,
        "correct": correct,
        "results": results_df,
    }


if __name__ == "__main__":
    models_dir = PROCESSED_DIR / "models"
    best_model_path = models_dir / "best_model.joblib"
    feature_cols_path = models_dir / "feature_columns.joblib"

    if not best_model_path.exists():
        logger.error("Best model not found. Run ensemble.py first.")
    elif not feature_cols_path.exists():
        logger.error("Feature columns not found. Run train.py first.")
    else:
        model = joblib.load(best_model_path)
        feature_cols = joblib.load(feature_cols_path)

        results = validate_against_live(model, feature_cols)