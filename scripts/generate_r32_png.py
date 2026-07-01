"""Generate a PNG table of all WC 2026 Round of 32 predictions."""

import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.features.build_2026_features import build_wc2026_features


def compute_standings():
    live = pd.read_csv(RAW_DIR / "wc2026_results_live.csv")
    live = live[live["home_score"].notna()].copy()
    live["home_score"] = live["home_score"].astype(int)
    live["away_score"] = live["away_score"].astype(int)
    gm = live[live["group"].notna() & (live["group"] != "R32")]

    standings = {}
    for grp in sorted(gm["group"].unique()):
        g = gm[gm["group"] == grp]
        teams = sorted(set(g["home_team"]).union(set(g["away_team"])))
        table = {}
        for t in teams:
            table[t] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
        for _, r in g.iterrows():
            h, a, hs, as_ = r["home_team"], r["away_team"], r["home_score"], r["away_score"]
            table[h]["P"] += 1
            table[a]["P"] += 1
            table[h]["GF"] += hs
            table[h]["GA"] += as_
            table[a]["GF"] += as_
            table[a]["GA"] += hs
            if hs > as_:
                table[h]["W"] += 1
                table[a]["L"] += 1
                table[h]["Pts"] += 3
            elif hs < as_:
                table[a]["W"] += 1
                table[h]["L"] += 1
                table[a]["Pts"] += 3
            else:
                table[h]["D"] += 1
                table[a]["D"] += 1
                table[h]["Pts"] += 1
                table[a]["Pts"] += 1
        df = pd.DataFrame(table).T
        df["GD"] = df["GF"] - df["GA"]
        standings[grp] = df.sort_values(["Pts", "GD", "GF"], ascending=False)
    return standings


def main():
    model = joblib.load(PROCESSED_DIR / "models" / "best_model.joblib")
    feature_cols = joblib.load(PROCESSED_DIR / "models" / "feature_columns.joblib")
    imputer = joblib.load(PROCESSED_DIR / "models" / "imputer.joblib")

    standings = compute_standings()
    W = {g: standings[g].index[0] for g in standings}
    R = {g: standings[g].index[1] for g in standings}
    T = {g: standings[g].index[2] for g in standings}

    # Third-place qualified groups: B, D, E, F, I, J, K, L
    # Official FIFA bracket for these groups:
    r32_matches = [
        (73, R["A"], R["B"]),       # 2A vs 2B
        (74, W["E"], T["D"]),       # 1E vs 3D
        (75, W["F"], R["C"]),       # 1F vs 2C
        (76, W["C"], R["F"]),       # 1C vs 2F
        (77, W["I"], T["F"]),       # 1I vs 3F  (France vs Sweden)
        (78, R["E"], R["I"]),       # 2E vs 2I
        (79, W["A"], T["E"]),       # 1A vs 3E
        (80, W["L"], T["K"]),       # 1L vs 3K
        (81, W["D"], T["B"]),       # 1D vs 3B
        (82, W["G"], T["I"]),       # 1G vs 3I
        (83, R["K"], R["L"]),       # 2K vs 2L
        (84, W["H"], R["J"]),       # 1H vs 2J
        (85, W["B"], T["J"]),       # 1B vs 3J
        (86, W["J"], R["H"]),       # 1J vs 2H
        (87, W["K"], T["L"]),       # 1K vs 3L
        (88, R["D"], R["G"]),       # 2D vs 2G
    ]

    fixtures_df = pd.DataFrame(
        [
            {"match_number": mn, "date": "2026-06-30", "home_team": h, "away_team": a, "group": "R32"}
            for mn, h, a in r32_matches
        ]
    )

    orig_path = PROCESSED_DIR / "wc2026_match_features.parquet"
    backup_path = PROCESSED_DIR / "wc2026_match_features_backup.parquet"
    shutil.copy2(orig_path, backup_path)

    r32_features = build_wc2026_features(fixtures_df=fixtures_df, include_live=True)

    shutil.copy2(backup_path, orig_path)
    backup_path.unlink()

    for col in feature_cols:
        if col not in r32_features.columns:
            r32_features[col] = 0.0

    X = r32_features[feature_cols].values
    if np.isnan(X).any():
        X = imputer.transform(X)

    proba = model.predict_proba(X)
    preds = model.predict(X)
    LABEL = ["away_win", "draw", "home_win"]

    # Check played
    live = pd.read_csv(RAW_DIR / "wc2026_results_live.csv")
    live = live[live["home_score"].notna()].copy()
    played_r32 = {}
    r32_live = live[live["group"].isna() | (live["group"] == "R32")]
    for _, r in r32_live.iterrows():
        played_r32[(r["home_team"], r["away_team"])] = (int(r["home_score"]), int(r["away_score"]))

    # Build table data
    col_labels = ["M#", "Home", "Away", "H%", "D%", "A%", "Pred", "Score", ""]
    cell_text = []
    cell_colors = []

    for i, (mn, h, a) in enumerate(r32_matches):
        p = proba[i]
        pr = preds[i]
        score = ""
        ok = ""
        key = (h, a)
        swap = (a, h)
        if key in played_r32:
            hs, as_ = played_r32[key]
            score = f"{hs}-{as_}"
            act = 2 if hs > as_ else (0 if hs < as_ else 1)
            ok = "Y" if pr == act else "N"
        elif swap in played_r32:
            hs, as_ = played_r32[swap]
            score = f"{hs}-{as_}"
            act = 2 if hs > as_ else (0 if hs < as_ else 1)
            ok = "Y" if pr == act else "N"

        cell_text.append([
            str(mn),
            h[:20],
            a[:20],
            f"{p[2]*100:.0f}",
            f"{p[1]*100:.0f}",
            f"{p[0]*100:.0f}",
            LABEL[pr],
            score,
            ok,
        ])
        rc = ["white"] * len(col_labels)
        if ok == "Y":
            rc[-1] = "#90EE90"
        elif ok == "N":
            rc[-1] = "#FFB6B6"
        if LABEL[pr] == "home_win":
            rc[6] = "#D6EAF8"
        elif LABEL[pr] == "away_win":
            rc[6] = "#FDEBD0"
        elif LABEL[pr] == "draw":
            rc[6] = "#FCF3CF"
        cell_colors.append(rc)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for i in range(len(col_labels)):
        table[0, i].set_facecolor("#2C3E50")
        table[0, i].set_text_props(color="white", fontweight="bold", fontsize=12)

    ax.set_title(
        "World Cup 2026 — Round of 32 Predictions",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    correct = sum(1 for r in cell_text if r[-1] == "Y")
    played_n = sum(1 for r in cell_text if r[-1] != "")
    pending = sum(1 for r in cell_text if r[-1] == "")

    fig.text(
        0.5,
        0.04,
        f"Played: {played_n}  |  Correct: {correct}  |  Pending: {pending}",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = PROCESSED_DIR / "r32_predictions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()