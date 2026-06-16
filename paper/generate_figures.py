#!/usr/bin/env python3
"""Generate publication-quality figures for the WC2026 paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Publication-quality settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
})

COLORS = {
    "Weighted Voting": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "Logistic Regression": "#2ca02c",
    "XGBoost": "#d62728",
    "Neural Network": "#9467bd",
    "home_win": "#2ecc71",
    "draw": "#3498db",
    "away_win": "#e74c3c",
}


def fig_model_comparison():
    """Figure 1: Model comparison (accuracy and log loss)."""
    models = ["Weighted\nVoting", "Random\nForest", "Logistic\nRegression", "XGBoost", "Neural\nNetwork"]
    accuracy = [0.615, 0.593, 0.579, 0.505, 0.359]
    log_loss = [0.835, 0.857, 0.874, 0.955, 1.732]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    x = np.arange(len(models))
    colors = [COLORS["Weighted Voting"], COLORS["Random Forest"],
              COLORS["Logistic Regression"], COLORS["XGBoost"], COLORS["Neural Network"]]

    bars = ax1.bar(x, accuracy, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylim(0.3, 0.7)
    ax1.axhline(y=0.333, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, accuracy):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, log_loss, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
    ax2.set_ylabel("Log Loss (cross-entropy)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylim(0.5, 1.5)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, log_loss):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_model_comparison.pdf")
    fig.savefig(OUT_DIR / "fig_model_comparison.png")
    plt.close(fig)
    print("Saved fig_model_comparison")


def fig_brier_scores():
    """Figure 2: Per-class Brier scores by model."""
    models = ["Weighted\nVoting", "Random\nForest", "Logistic\nRegression", "XGBoost", "Neural\nNetwork"]
    brier_home = [0.174, 0.181, 0.189, 0.194, 0.304]
    brier_draw = [0.172, 0.177, 0.181, 0.227, 0.427]
    brier_away = [0.144, 0.147, 0.146, 0.160, 0.222]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, brier_home, width, label="Home win", color=COLORS["home_win"],
           edgecolor="black", linewidth=0.5)
    ax.bar(x, brier_draw, width, label="Draw", color=COLORS["draw"],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width, brier_away, width, label="Away win", color=COLORS["away_win"],
           edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Brier Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.42)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_brier_scores.pdf")
    fig.savefig(OUT_DIR / "fig_brier_scores.png")
    plt.close(fig)
    print("Saved fig_brier_scores")


def fig_live_validation():
    """Figure 3: Live validation predicted probabilities for actual draws."""
    matches = [
        "CAN-BIH", "QAT-SUI", "BRA-MAR", "NED-JPN", "ESP-CPV",
        "MEX-RSA", "KOR-CZE", "USA-PAR", "HAI-SCO", "GER-CUW",
        "AUS-TUR", "CIV-ECU", "SWE-TUN",
    ]
    home_probs = [0.69, 0.10, 0.46, 0.45, 0.78, 0.82, 0.40, 0.47, 0.17, 0.78, 0.22, 0.30, 0.54]
    draw_probs = [0.20, 0.16, 0.28, 0.29, 0.14, 0.12, 0.29, 0.24, 0.22, 0.14, 0.24, 0.28, 0.22]
    away_probs = [0.11, 0.75, 0.25, 0.26, 0.08, 0.06, 0.31, 0.29, 0.61, 0.08, 0.55, 0.42, 0.25]
    actual_draw = [True, True, True, True, True, False, False, False, False, False, False, False, False]

    y = np.arange(len(matches))
    height = 0.8

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_h = ax.barh(y, home_probs, height, color=COLORS["home_win"], edgecolor="black", linewidth=0.3, label="P(home win)")
    bars_d = ax.barh(y, draw_probs, height, left=home_probs, color=COLORS["draw"], edgecolor="black", linewidth=0.3, label="P(draw)")
    bars_a = ax.barh(y, away_probs, height, left=[h + d for h, d in zip(home_probs, draw_probs)],
                     color=COLORS["away_win"], edgecolor="black", linewidth=0.3, label="P(away win)")

    for i, is_draw in enumerate(actual_draw):
        if is_draw:
            ax.annotate("DRAW", xy=(0.5, i), fontsize=7, fontweight="bold",
                        color="white", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))

    ax.set_yticks(y)
    ax.set_yticklabels(matches, fontsize=9)
    ax.set_xlabel("Predicted Probability")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_live_validation.pdf")
    fig.savefig(OUT_DIR / "fig_live_validation.png")
    plt.close(fig)
    print("Saved fig_live_validation")


def fig_tournament_probabilities():
    """Figure 4: Tournament advancement probabilities."""
    tp = pd.read_csv(DATA_DIR / "tournament_probabilities.csv")
    top20 = tp.head(20).sort_values("prob_winner", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y = np.arange(len(top20))
    width = 0.2

    ax.barh(y + 1.5 * width, top20["prob_ro32"], width, label="R32", color="#a8d5e2", edgecolor="black", linewidth=0.3)
    ax.barh(y + 0.5 * width, top20["prob_ro16"], width, label="Ro16", color="#4a90d9", edgecolor="black", linewidth=0.3)
    ax.barh(y - 0.5 * width, top20["prob_qf"], width, label="QF", color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.barh(y - 1.5 * width, top20["prob_sf"], width, label="SF", color="#0d4f8b", edgecolor="black", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(top20["team"], fontsize=9)
    ax.set_xlabel("Probability")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 1.15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tournament_probs.pdf")
    fig.savefig(OUT_DIR / "fig_tournament_probs.png")
    plt.close(fig)
    print("Saved fig_tournament_probs")


def fig_confusion_draw():
    """Figure 5: Draw prediction analysis - predicted vs actual outcomes."""
    categories = ["Actual\nHome Win", "Actual\nDraw", "Actual\nAway Win"]
    predicted_home = [6, 5, 1]
    predicted_draw = [0, 0, 0]
    predicted_away = [1, 0, 0]

    # More detailed: per-match D/Max ratio for actual draws vs non-draws
    draws_dmax = [0.29, 0.21, 0.61, 0.64, 0.18]
    nondraws_dmax = [0.14, 0.73, 0.52, 0.37, 0.17, 0.43, 0.41, 0.66]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: confusion-style bar chart
    x = np.arange(3)
    width = 0.25
    ax1.bar(x - width, predicted_home, width, label="Predicted Home", color=COLORS["home_win"], edgecolor="black", linewidth=0.5)
    ax1.bar(x, predicted_draw, width, label="Predicted Draw", color=COLORS["draw"], edgecolor="black", linewidth=0.5)
    ax1.bar(x + width, predicted_away, width, label="Predicted Away", color=COLORS["away_win"], edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("Number of Matches")
    ax1.set_title("Predicted vs Actual Outcomes (n=13)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right: D/Max ratio distribution
    ax2.hist(nondraws_dmax, bins=8, range=(0, 1), alpha=0.7, color="#95a5a6",
             edgecolor="black", linewidth=0.5, label="Non-draw matches")
    ax2.hist(draws_dmax, bins=8, range=(0, 1), alpha=0.7, color=COLORS["draw"],
             edgecolor="black", linewidth=0.5, label="Draw matches")
    ax2.axvline(x=0.85, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.85")
    ax2.set_xlabel("Draw Ratio (D/Max)")
    ax2.set_ylabel("Number of Matches")
    ax2.set_title("Draw Ratio Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_draw_analysis.pdf")
    fig.savefig(OUT_DIR / "fig_draw_analysis.png")
    plt.close(fig)
    print("Saved fig_draw_analysis")


def fig_elo_rating_distribution():
    """Figure 6: Elo rating distribution of WC 2026 teams."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    elo_df = pd.read_parquet(DATA_DIR / "elo_ratings_current.parquet")

    # Get WC 2026 teams - file has columns: group, team, pot
    groups_df = pd.read_csv(DATA_DIR.parent / "raw" / "wc2026_groups.csv")
    wc_teams = set(groups_df["team"].dropna().tolist())

    # If elo file is incomplete (only a few teams), rebuild it
    if len(elo_df) < 40:
        from src.features.elo import EloRatingSystem
        from src.config import RAW_DIR
        matches = pd.read_csv(RAW_DIR / "international_matches" / "results.csv")
        elo_sys = EloRatingSystem()
        elo_df = elo_sys.compute_elo_ratings(matches)
        print(f"  Rebuilt elo ratings: {len(elo_df)} teams")

    wc_elo = elo_df[elo_df["team"].isin(wc_teams)].sort_values("elo", ascending=False)

    if len(wc_elo) == 0:
        print(f"  WARNING: No WC teams found in elo data. Teams in groups: {len(wc_teams)}")
        print(f"  Sample elo teams: {elo_df['team'].head(10).tolist()}")
        print(f"  Sample group teams: {list(wc_teams)[:10]}")
        # Try normalizing team names
        from src.helpers import normalize_team_name
        wc_teams_norm = {normalize_team_name(t) for t in wc_teams}
        elo_teams_norm = {normalize_team_name(t) for t in elo_df["team"]}
        overlap = wc_teams_norm & elo_teams_norm
        print(f"  Overlap after normalization: {len(overlap)}")
        wc_elo = elo_df[elo_df["team"].apply(normalize_team_name).isin(wc_teams_norm)].sort_values("elo", ascending=False)

    print(f"  WC teams in elo: {len(wc_elo)}/{len(wc_teams)}")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y = np.arange(len(wc_elo))
    colors_list = []
    host_nations = {"Mexico", "United States", "Canada"}
    for _, row in wc_elo.iterrows():
        if row["team"] in host_nations:
            colors_list.append("#e74c3c")
        else:
            colors_list.append("#3498db")

    ax.barh(y, wc_elo["elo"], color=colors_list, edgecolor="black", linewidth=0.3, height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(wc_elo["team"], fontsize=7)
    ax.set_xlabel("Elo Rating")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="Host nations"),
        Patch(facecolor="#3498db", edgecolor="black", label="Other teams"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_elo_ratings.pdf")
    fig.savefig(OUT_DIR / "fig_elo_ratings.png")
    plt.close(fig)
    print("Saved fig_elo_ratings")


def fig_feature_importance():
    """Figure 7: Top 20 feature importances from XGBoost."""
    import joblib
    model = joblib.load(DATA_DIR / "models" / "xgboost.joblib")
    feature_cols = joblib.load(DATA_DIR / "models" / "feature_columns.joblib")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(indices))
    ax.barh(y, importances[indices][::-1], color="#3498db", edgecolor="black", linewidth=0.3, height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_cols[i] for i in indices[::-1]], fontsize=8)
    ax.set_xlabel("Feature Importance (gain)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_feature_importance.pdf")
    fig.savefig(OUT_DIR / "fig_feature_importance.png")
    plt.close(fig)
    print("Saved fig_feature_importance")


def fig_calibration():
    """Figure 8: Calibration curves for each model."""
    import joblib

    models_dict = {
        "Stacking Ensemble": joblib.load(DATA_DIR / "models" / "best_model.joblib"),
        "XGBoost": joblib.load(DATA_DIR / "models" / "xgboost.joblib"),
        "Random Forest": joblib.load(DATA_DIR / "models" / "randomforest.joblib"),
    }
    feature_cols = joblib.load(DATA_DIR / "models" / "feature_columns.joblib")
    imputer = joblib.load(DATA_DIR / "models" / "imputer.joblib")

    from src.models.train import split_data, LABEL_MAP
    result = split_data(pd.read_parquet(DATA_DIR / "match_features.parquet"))
    X_train, y_train, X_val, y_val, X_test, y_test = result[:6]
    feature_cols = result[6]

    fig, ax = plt.subplots(figsize=(6, 6))

    from sklearn.calibration import calibration_curve

    for name, model in models_dict.items():
        try:
            X_imp = imputer.transform(X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_cols))
            y_proba = model.predict_proba(X_imp)
            prob_draw = y_proba[:, 1]
            y_draw = (y_test == 1).astype(int)  # draw is label 1

            fraction_pos, mean_pred = calibration_curve(y_draw, prob_draw, n_bins=10, strategy="uniform")
            ax.plot(mean_pred, fraction_pos, marker="o", linewidth=1.5, label=name, markersize=4)
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Draw Class)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_calibration.pdf")
    fig.savefig(OUT_DIR / "fig_calibration.png")
    plt.close(fig)
    print("Saved fig_calibration")


if __name__ == "__main__":
    print("Generating figures...")
    fig_model_comparison()
    fig_brier_scores()
    fig_live_validation()
    fig_tournament_probabilities()
    fig_confusion_draw()
    fig_elo_rating_distribution()
    fig_feature_importance()
    fig_calibration()
    print("Done! All figures generated.")