"""Generate a PNG table of all WC 2026 group stage predictions."""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


def main():
    model = joblib.load(PROCESSED_DIR / "models" / "best_model.joblib")
    feature_cols = joblib.load(PROCESSED_DIR / "models" / "feature_columns.joblib")
    imputer = joblib.load(PROCESSED_DIR / "models" / "imputer.joblib")

    wc = pd.read_parquet(PROCESSED_DIR / "wc2026_match_features.parquet")
    live = pd.read_csv(RAW_DIR / "wc2026_results_live.csv")
    played = {}
    for _, r in live.iterrows():
        if pd.notna(r["home_score"]) and pd.notna(r["away_score"]):
            played[(r["home_team"], r["away_team"])] = (int(r["home_score"]), int(r["away_score"]))

    LABEL = ["away_win", "draw", "home_win"]

    for col in feature_cols:
        if col not in wc.columns:
            wc[col] = 0.0

    X = wc[feature_cols].values
    if np.isnan(X).any():
        X = imputer.transform(X)

    proba = model.predict_proba(X)
    preds = model.predict(X)

    # Build row data
    rows = []
    for idx, (_, row) in enumerate(wc.iterrows()):
        h, a = row["home_team"], row["away_team"]
        p = proba[idx].copy()
        pr = preds[idx]
        key = (h, a)
        swap_key = (a, h)
        actual = ""
        ok = ""

        if key in played:
            hs, as_ = played[key]
            actual = f"{hs}-{as_}"
            act_lbl = 2 if hs > as_ else (0 if hs < as_ else 1)
            ok = "Y" if pr == act_lbl else "N"
        elif swap_key in played:
            hs, as_ = played[swap_key]
            p = np.array([p[2], p[1], p[0]])
            pr = {0: 2, 1: 1, 2: 0}[pr]
            actual = f"{hs}-{as_}"
            act_lbl = 2 if hs > as_ else (0 if hs < as_ else 1)
            ok = "Y" if pr == act_lbl else "N"

        rows.append({
            "group": row["group"],
            "match": int(row["match_number"]),
            "home": h,
            "away": a,
            "H": p[2],
            "D": p[1],
            "A": p[0],
            "pred": LABEL[pr],
            "score": actual,
            "ok": ok,
        })

    df = pd.DataFrame(rows)

    n_groups = df["group"].nunique()
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 22), dpi=150)
    axes = axes.flatten()

    groups = sorted(df["group"].unique())
    for gi, group in enumerate(groups):
        ax = axes[gi]
        gdf = df[df["group"] == group].sort_values("match")

        col_labels = ["M", "Home", "Away", "H%", "D%", "A%", "Pred", "Score", ""]
        cell_text = []
        cell_colors = []
        for _, r in gdf.iterrows():
            cell_text.append([
                str(r["match"]),
                r["home"][:18],
                r["away"][:18],
                f"{r['H']*100:.0f}",
                f"{r['D']*100:.0f}",
                f"{r['A']*100:.0f}",
                r["pred"],
                r["score"],
                r["ok"],
            ])
            row_colors = ["white"] * len(col_labels)
            if r["ok"] == "Y":
                row_colors[-1] = "#90EE90"
            elif r["ok"] == "N":
                row_colors[-1] = "#FFB6B6"
            if r["pred"] == "home_win":
                row_colors[6] = "#D6EAF8"
            elif r["pred"] == "away_win":
                row_colors[6] = "#FDEBD0"
            elif r["pred"] == "draw":
                row_colors[6] = "#FCF3CF"
            cell_colors.append(row_colors)

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            cellLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        for i in range(len(col_labels)):
            table[0, i].set_facecolor("#34495E")
            table[0, i].set_text_props(color="white", fontweight="bold")

        ax.set_title(f"GROUP {group}", fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")

    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    correct = (df["ok"] == "Y").sum()
    played_n = (df["ok"] != "").sum()
    wrong = (df["ok"] == "N").sum()
    pending = (df["ok"] == "").sum()

    fig.suptitle(
        f"World Cup 2026 — Full Group Stage Predictions\n"
        f"Played: {played_n}  |  Correct: {correct}  |  Wrong: {wrong}  |  Pending: {pending}  |  Accuracy: {correct/max(played_n,1)*100:.0f}%",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PROCESSED_DIR / "group_stage_predictions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()