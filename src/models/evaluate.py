import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)

from src.config import PROCESSED_DIR, RANDOM_STATE
from src.helpers import logger

warnings.filterwarnings("ignore", category=UserWarning)

LABEL_NAMES = ["away_win", "draw", "home_win"]


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    logger.info(f"Evaluating {model_name}...")

    y_test = np.asarray(y_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)

    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    brier_scores = {}
    for i, label in enumerate(LABEL_NAMES):
        y_true_binary = (y_test == i).astype(int)
        brier_scores[label] = brier_score_loss(y_true_binary, y_prob[:, i])

    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "log_loss": ll,
        "brier_scores": brier_scores,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
    }

    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Log Loss: {ll:.4f}")
    logger.info(f"  Brier Scores: {brier_scores}")

    return results


def compare_models(results_dict: dict[str, dict]) -> pd.DataFrame:
    logger.info("Comparing models...")

    comparison = []
    for name, results in results_dict.items():
        comparison.append(
            {
                "Model": name,
                "Accuracy": results["accuracy"],
                "Log Loss": results["log_loss"],
                "Brier (home_win)": results["brier_scores"].get("home_win", np.nan),
                "Brier (draw)": results["brier_scores"].get("draw", np.nan),
                "Brier (away_win)": results["brier_scores"].get("away_win", np.nan),
                "Avg Brier": np.mean(list(results["brier_scores"].values())),
            }
        )

    df = pd.DataFrame(comparison)
    df = df.sort_values("Log Loss")

    logger.info(f"\n{df.to_string(index=False)}")

    return df


def _is_interactive():
    try:
        get_ipython()
        return True
    except NameError:
        return False


def plot_feature_importance(model, feature_cols: list[str], top_n: int = 20):
    logger.info("Plotting feature importance...")

    try:
        import matplotlib
        if not _is_interactive():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps") and "xgbclassifier" in model.named_steps:
            importances = model.named_steps["xgbclassifier"].feature_importances_
        elif hasattr(model, "estimators_"):
            for name, est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances = est.feature_importances_
                    break
            else:
                logger.info("No feature_importances_ found in ensemble")
                return
        else:
            logger.info(f"Cannot extract feature importance from {type(model)}")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_cols[i] for i in indices if i < len(feature_cols)]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_importances[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top Feature Importances")
        plt.tight_layout()

        out_path = PROCESSED_DIR / "evaluation" / "feature_importance.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if not _is_interactive():
            plt.close()
        logger.info(f"  Saved to {out_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


def plot_calibration_curves(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray):
    logger.info("Plotting calibration curves...")

    try:
        import matplotlib
        if not _is_interactive():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        class_names = ["Home Win", "Draw", "Away Win"]

        y_test_arr = np.asarray(y_test)

        for class_idx, class_name in enumerate(class_names):
            ax = axes[class_idx]
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

            for name, results in models_dict.items():
                y_prob = results["y_prob"][:, class_idx]
                y_true_binary = (y_test_arr == class_idx).astype(int)

                fraction_pos, mean_predicted = calibration_curve(
                    y_true_binary, y_prob, n_bins=10, strategy="uniform"
                )

                ax.plot(mean_predicted, fraction_pos, marker="o", label=name)

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f"Calibration: {class_name}")
            ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        out_path = PROCESSED_DIR / "evaluation" / "calibration_curves.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if not _is_interactive():
            plt.close()
        logger.info(f"  Saved to {out_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


def generate_evaluation_report(all_results: dict[str, dict], feature_cols: list[str] | None = None):
    logger.info("Generating evaluation report...")

    eval_dir = PROCESSED_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_models(all_results)
    comparison.to_csv(eval_dir / "model_comparison.csv", index=False)

    if feature_cols:
        best_model_name = comparison.iloc[0]["Model"]
        best_result = all_results[best_model_name]
        if hasattr(best_result.get("model"), "feature_importances_"):
            plot_feature_importance(best_result["model"], feature_cols)

    plot_calibration_curves(all_results, None, all_results[list(all_results.keys())[0]]["y_test"] if all_results else None)

    report = {
        "model_comparison": comparison.to_dict(),
        "best_model": comparison.iloc[0]["Model"],
        "best_accuracy": float(comparison.iloc[0]["Accuracy"]),
        "best_log_loss": float(comparison.iloc[0]["Log Loss"]),
    }

    import json
    with open(eval_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Evaluation report saved to {eval_dir}")

    return comparison


if __name__ == "__main__":
    models_dir = PROCESSED_DIR / "models"

    features_path = PROCESSED_DIR / "match_features.parquet"
    if not features_path.exists():
        logger.error("Features not found. Run build_features.py first.")
    else:
        df = pd.read_parquet(features_path)
        df = df.dropna(subset=["outcome"])

        from src.models.train import split_data

        (X_train, y_train, X_val, y_val, X_test, y_test,
         feature_cols, train_df, val_df, test_df) = split_data(df)

        all_results = {}
        for model_file in models_dir.glob("*.joblib"):
            if model_file.stem in ("feature_columns", "best_model"):
                continue

            model = joblib.load(model_file)
            name = model_file.stem.replace("_", " ").title()
            results = evaluate_model(model, X_test, y_test, name)
            results["model"] = model
            all_results[name] = results

        if all_results:
            generate_evaluation_report(all_results, feature_cols)