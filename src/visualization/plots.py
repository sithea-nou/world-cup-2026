import numpy as np
import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DIR
from src.helpers import logger


def _is_interactive():
    try:
        get_ipython()
        return True
    except NameError:
        return False


def _setup_plot_style():
    import matplotlib
    if not _is_interactive():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (14, 8),
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
        }
    )
    return plt


def plot_tournament_probabilities(df: pd.DataFrame, top_n: int = 20, save: bool = True):
    logger.info("Plotting tournament probabilities...")
    plt = _setup_plot_style()

    df = df.sort_values("prob_winner", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))

    bars = ax.barh(df["team"], df["prob_winner"] * 100, color=colors)

    for bar, val in zip(bars, df["prob_winner"] * 100):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Winning Probability (%)")
    ax.set_title(f"World Cup 2026 - Top {top_n} Winning Probabilities")
    ax.set_xlim(0, df["prob_winner"].max() * 100 * 1.3)

    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "tournament_probabilities.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig


def plot_group_heatmaps(group_probs: pd.DataFrame, save: bool = True):
    logger.info("Plotting group stage heatmaps...")
    plt = _setup_plot_style()
    import seaborn as sns

    groups = group_probs["group"].unique()
    n_groups = len(groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, group in enumerate(sorted(groups)):
        if idx >= len(axes):
            break

        ax = axes[idx]
        group_data = group_probs[group_probs["group"] == group].sort_values("prob_advance", ascending=False)

        positions = ["prob_1st", "prob_2nd", "prob_3rd", "prob_4th"]
        heatmap_data = group_data[positions].values

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            xticklabels=["1st", "2nd", "3rd", "4th"],
            yticklabels=group_data["team"].values,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            cbar=False,
        )
        ax.set_title(f"Group {group}")

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("World Cup 2026 - Group Stage Advancement Probabilities", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "group_heatmaps.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig


def plot_feature_importance(model, feature_cols: list[str], top_n: int = 20, save: bool = True):
    logger.info("Plotting feature importance...")
    plt = _setup_plot_style()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        for name, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                break
        else:
            logger.info("No feature_importances_ found")
            return None
    else:
        logger.info(f"Cannot extract feature importance from {type(model)}")
        return None

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in indices if i < len(feature_cols)]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_importances[::-1], color=plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "feature_importance.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig


def plot_elo_ratings(ratings_df: pd.DataFrame, top_n: int = 30, save: bool = True):
    logger.info("Plotting Elo ratings...")
    plt = _setup_plot_style()

    top_teams = ratings_df.sort_values("elo", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top_teams)))
    bars = ax.barh(top_teams["team"], top_teams["elo"], color=colors)

    ax.set_xlabel("Elo Rating")
    ax.set_title(f"Top {top_n} Teams by Elo Rating")
    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "elo_ratings.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, save: bool = True):
    logger.info("Plotting model comparison...")
    plt = _setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    x = range(len(comparison_df))
    width = 0.6
    ax1.bar(x, comparison_df["Accuracy"], width, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(comparison_df))))
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy Comparison")

    ax2 = axes[1]
    ax2.bar(x, comparison_df["Log Loss"], width, color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(comparison_df))))
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
    ax2.set_ylabel("Log Loss (lower is better)")
    ax2.set_title("Model Log Loss Comparison")

    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "model_comparison.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig


def plot_match_predictions(predictions: pd.DataFrame, save: bool = True):
    logger.info("Plotting match predictions...")
    plt = _setup_plot_style()

    groups = sorted(predictions["group"].unique()) if "group" in predictions.columns and predictions["group"].notna().any() else [""]
    n_groups = len(groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, group in enumerate(groups):
        if idx >= len(axes):
            break

        ax = axes[idx]
        if group:
            group_data = predictions[predictions["group"] == group].reset_index(drop=True)
        else:
            group_data = predictions.reset_index(drop=True)

        if group_data.empty:
            ax.set_visible(False)
            continue

        labels = [f"{r['home_team']} vs {r['away_team']}" for _, r in group_data.iterrows()]
        y_pos = np.arange(len(group_data))

        ax.barh(y_pos, group_data["prob_home_win"], color="#2ecc71", label="Home Win")
        ax.barh(y_pos, group_data["prob_draw"], left=group_data["prob_home_win"], color="#3498db", label="Draw")
        ax.barh(
            y_pos, group_data["prob_away_win"],
            left=group_data["prob_home_win"] + group_data["prob_draw"],
            color="#e74c3c", label="Away Win",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Probability")
        title = f"Group {group}" if group else "All Matches"
        ax.set_title(title, fontsize=11)

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper right", fontsize=10)
    plt.suptitle("World Cup 2026 - Per-Match Predictions", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "match_predictions.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig


def plot_round_probabilities(df: pd.DataFrame, save: bool = True):
    logger.info("Plotting round advancement probabilities...")
    plt = _setup_plot_style()

    top_16 = df.sort_values("prob_winner", ascending=False).head(16)

    rounds = ["prob_ro32", "prob_ro16", "prob_qf", "prob_sf", "prob_final", "prob_winner"]
    round_labels = ["Ro32", "Ro16", "QF", "SF", "Final", "Winner"]

    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(top_16))
    width = 0.13

    for i, (round_col, label) in enumerate(zip(rounds, round_labels)):
        offset = (i - len(rounds) / 2) * width
        ax.bar(x + offset, top_16[round_col] * 100, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(top_16["team"], rotation=45, ha="right")
    ax.set_ylabel("Probability (%)")
    ax.set_title("World Cup 2026 - Round Advancement Probabilities (Top 16)")
    ax.legend()
    plt.tight_layout()

    if save:
        out_path = PROCESSED_DIR / "evaluation" / "round_probabilities.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved to {out_path}")

    if not _is_interactive():
        plt.close()
    return fig