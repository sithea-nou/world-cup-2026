import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from src.config import PROCESSED_DIR, RAW_DIR
from src.features.build_features import build_match_features
from src.features.elo import EloRatingSystem
from src.helpers import normalize_team_name
from src.models.train import _get_feature_columns, _compute_sample_weights
import joblib


def backtest_wc2022():
    print("=" * 60)
    print("WC 2022 BACKTEST")
    print("=" * 60)

    df = pd.read_csv(RAW_DIR / "international_matches" / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    wc2022 = df[
        (df["tournament"].str.contains("World Cup", case=False, na=False))
        & (~df["tournament"].str.contains("qualification|qualifying", case=False, na=False))
        & (df["date"].dt.year == 2022)
        & (df["home_score"].notna())
        & (df["away_score"].notna())
    ].copy()
    wc2022 = wc2022.sort_values("date")

    group_matches = wc2022.head(48).copy()
    knockout_matches = wc2022.iloc[48:].copy()

    print(f"\nGroup stage: {len(group_matches)} matches")
    print(f"Knockout stage: {len(knockout_matches)} matches")

    group_outcomes = group_matches.apply(
        lambda r: 2 if r["home_score"] > r["away_score"] else (0 if r["home_score"] < r["away_score"] else 1),
        axis=1,
    )
    print(f"Group outcomes: home_win={sum(group_outcomes==2)}, draw={sum(group_outcomes==1)}, away_win={sum(group_outcomes==0)}")
    print(f"Group draw rate: {sum(group_outcomes==1)/len(group_outcomes)*100:.1f}%")

    all_outcomes = wc2022.apply(
        lambda r: 2 if r["home_score"] > r["away_score"] else (0 if r["home_score"] < r["away_score"] else 1),
        axis=1,
    )
    print(f"\nAll WC2022 outcomes: home_win={sum(all_outcomes==2)}, draw={sum(all_outcomes==1)}, away_win={sum(all_outcomes==0)}")

    features_df = build_match_features(include_live=False)

    wc2022_date_start = pd.Timestamp("2022-11-20")
    wc2022_date_end = pd.Timestamp("2022-12-18")

    wc2022_features = features_df[
        (features_df["date"] >= wc2022_date_start)
        & (features_df["date"] <= wc2022_date_end)
        & (features_df["is_world_cup"] == 1)
    ].copy()

    print(f"\nWC 2022 features found: {len(wc2022_features)} matches")

    if len(wc2022_features) == 0:
        print("No WC 2022 features found in dataset. Features may not cover this period.")
        return

    model_path = PROCESSED_DIR / "models" / "best_model.joblib"
    feature_cols_path = PROCESSED_DIR / "models" / "feature_columns.joblib"
    imputer_path = PROCESSED_DIR / "models" / "imputer.joblib"

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)
    imputer = joblib.load(imputer_path)

    available_cols = [c for c in feature_cols if c in wc2022_features.columns]
    missing_cols = [c for c in feature_cols if c not in wc2022_features.columns]
    print(f"Available feature cols: {len(available_cols)}/{len(feature_cols)}")
    if missing_cols:
        print(f"Missing cols: {missing_cols}")

    for col in missing_cols:
        wc2022_features[col] = 0.0

    results = []
    label_names = {0: "away_win", 1: "draw", 2: "home_win"}

    for idx, row in wc2022_features.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        hs = row.get("home_score", np.nan)
        as_ = row.get("away_score", np.nan)

        if pd.isna(hs) or pd.isna(as_):
            continue

        if hs > as_:
            actual = 2
        elif hs < as_:
            actual = 0
        else:
            actual = 1

        X = row[feature_cols].values.astype(float).flatten().reshape(1, -1)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = imputer.transform(X)

        proba = model.predict_proba(X)[0]
        pred_class = model.predict(X)[0]

        is_correct = pred_class == actual
        actual_prob = proba[actual]

        is_group = row.get("is_knockout", 0) == 0

        results.append(
            {
                "date": row["date"],
                "home_team": home,
                "home_score": int(hs),
                "away_team": away,
                "away_score": int(as_),
                "actual": label_names[actual],
                "predicted": label_names[pred_class],
                "prob_home_win": proba[2],
                "prob_draw": proba[1],
                "prob_away_win": proba[0],
                "correct": is_correct,
                "actual_prob": actual_prob,
                "is_group": is_group,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results to evaluate")
        return

    print(f"\nTotal matches evaluated: {len(results_df)}")

    group_df = results_df[results_df["is_group"]]
    knockout_df = results_df[~results_df["is_group"]]

    for label, subset in [("Group Stage", group_df), ("Knockout Stage", knockout_df), ("All", results_df)]:
        if subset.empty:
            continue
        acc = accuracy_score(
            [2 if a == "home_win" else (0 if a == "away_win" else 1) for a in subset["actual"]],
            [2 if p == "home_win" else (0 if p == "away_win" else 1) for p in subset["predicted"]],
        )
        y_true = [2 if a == "home_win" else (0 if a == "away_win" else 1) for a in subset["actual"]]
        y_prob = subset[["prob_away_win", "prob_draw", "prob_home_win"]].values
        ll = log_loss(y_true, y_prob, labels=[0, 1, 2])

        draw_mask = subset["actual"] == "draw"
        draw_pred = subset["predicted"] == "draw"

        print(f"\n{label} ({len(subset)} matches):")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Log Loss: {ll:.4f}")
        print(f"  Actual draws: {draw_mask.sum()}")
        print(f"  Predicted draws: {draw_pred.sum()}")
        print(f"  Draw recall: {(draw_mask & draw_pred).sum()}/{draw_mask.sum()}")

    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    correct_count = results_df["correct"].sum()
    total = len(results_df)
    print(f"\nOverall: {correct_count}/{total} correct ({correct_count/total*100:.1f}%)")

    print("\nINCORRECT PREDICTIONS:")
    wrong = results_df[~results_df["correct"]]
    for _, r in wrong.iterrows():
        marker = "DRAW-MISS" if r["actual"] == "draw" else "UPSET"
        print(f"  [{marker}] {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']} | Pred: {r['predicted']} | Actual: {r['actual']} | H:{r['prob_home_win']:.2f} D:{r['prob_draw']:.2f} A:{r['prob_away_win']:.2f}")

    print("\nCORRECT PREDICTIONS:")
    correct = results_df[results_df["correct"]]
    for _, r in correct.iterrows():
        print(f"  {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']} | Pred: {r['predicted']} | H:{r['prob_home_win']:.2f} D:{r['prob_draw']:.2f} A:{r['prob_away_win']:.2f}")


if __name__ == "__main__":
    backtest_wc2022()