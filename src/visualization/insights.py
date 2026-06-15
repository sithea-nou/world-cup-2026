import numpy as np
import pandas as pd

from src.config import CONFEDERATIONS, HOST_NATIONS, PROCESSED_DIR
from src.helpers import logger


def compute_confederation_stats(tournament_probs: pd.DataFrame) -> pd.DataFrame:
    confed_map = CONFEDERATIONS
    stats = []
    for confed in sorted(set(confed_map.values())):
        teams = [t for t in tournament_probs["team"] if confed_map.get(t) == confed]
        if not teams:
            continue
        confed_data = tournament_probs[tournament_probs["team"].isin(teams)]
        stats.append(
            {
                "confederation": confed,
                "n_teams": len(teams),
                "avg_prob_winner": confed_data["prob_winner"].mean() * 100,
                "avg_prob_advance_ro16": confed_data["prob_ro16"].mean() * 100,
                "avg_prob_qf": confed_data["prob_qf"].mean() * 100,
                "best_team": confed_data.loc[confed_data["prob_winner"].idxmax(), "team"],
                "best_team_win_pct": confed_data["prob_winner"].max() * 100,
            }
        )

    df = pd.DataFrame(stats).sort_values("avg_prob_winner", ascending=False).reset_index(drop=True)
    return df


def compute_host_nation_impact(tournament_probs: pd.DataFrame, match_preds: pd.DataFrame) -> dict:
    host_data = {}
    for host in HOST_NATIONS:
        tp = tournament_probs[tournament_probs["team"] == host]
        mp = match_preds[(match_preds["home_team"] == host) | (match_preds["away_team"] == host)]

        home_matches = mp[mp["home_team"] == host]
        away_matches = mp[mp["away_team"] == host]

        host_data[host] = {
            "prob_winner": tp["prob_winner"].values[0] * 100 if not tp.empty else 0,
            "prob_final": tp["prob_final"].values[0] * 100 if not tp.empty else 0,
            "prob_sf": tp["prob_sf"].values[0] * 100 if not tp.empty else 0,
            "prob_qf": tp["prob_qf"].values[0] * 100 if not tp.empty else 0,
            "prob_ro16": tp["prob_ro16"].values[0] * 100 if not tp.empty else 0,
            "avg_home_win_pct": home_matches["prob_home_win"].mean() * 100 if not home_matches.empty else 0,
            "avg_draw_pct_when_home": home_matches["prob_draw"].mean() * 100 if not home_matches.empty else 0,
            "avg_away_win_pct_when_away": away_matches["prob_away_win"].mean() * 100 if not away_matches.empty else 0,
            "home_matches_predicted_wins": (home_matches["prediction"] == "home_win").sum(),
            "home_matches_total": len(home_matches),
        }

    return host_data


def compute_draw_analysis(match_preds: pd.DataFrame) -> dict:
    total = len(match_preds)
    draws = (match_preds["prediction"] == "draw").sum()
    home_wins = (match_preds["prediction"] == "home_win").sum()
    away_wins = (match_preds["prediction"] == "away_win").sum()

    high_draw = match_preds[match_preds["prob_draw"] > 0.40].sort_values("prob_draw", ascending=False)

    avg_draw_pct = match_preds["prob_draw"].mean() * 100
    max_draw_pct = match_preds["prob_draw"].max() * 100
    min_draw_pct = match_preds["prob_draw"].min() * 100

    return {
        "total_matches": total,
        "predicted_draws": int(draws),
        "predicted_home_wins": int(home_wins),
        "predicted_away_wins": int(away_wins),
        "draw_rate": draws / total * 100,
        "avg_draw_prob": avg_draw_pct,
        "max_draw_prob": max_draw_pct,
        "min_draw_prob": min_draw_pct,
        "high_draw_matches": high_draw[["group", "home_team", "away_team", "prob_draw"]].head(10),
    }


def compute_dark_horses(tournament_probs: pd.DataFrame, group_probs: pd.DataFrame) -> pd.DataFrame:
    confed_map = CONFEDERATIONS
    non_uefa_conmebol = [
        t for t in tournament_probs["team"]
        if confed_map.get(t) not in ("UEFA", "CONMEBOL")
    ]

    if not non_uefa_conmebol:
        return pd.DataFrame()

    dark_horses = tournament_probs[tournament_probs["team"].isin(non_uefa_conmebol)].copy()
    dark_horses = dark_horses.sort_values("prob_winner", ascending=False).head(10)

    if not group_probs.empty:
        advance_map = group_probs.set_index("team")["prob_advance"].to_dict()
        dark_horses["prob_advance"] = dark_horses["team"].map(advance_map).fillna(0) * 100
        pos_map = group_probs.set_index("team")["prob_1st"].to_dict()
        dark_horses["prob_1st_in_group"] = dark_horses["team"].map(pos_map).fillna(0) * 100

    return dark_horses


def compare_with_odds(tournament_probs: pd.DataFrame) -> pd.DataFrame | None:
    odds_path = PROCESSED_DIR / ".." / "raw" / "odds_outright.csv"
    if not odds_path.exists():
        odds_path = PROCESSED_DIR / "raw" / "odds_outright.csv"
    import os
    for candidate in [
        PROCESSED_DIR.parent / "raw" / "odds_outright.csv",
        PROCESSED_DIR / "odds_outright.csv",
    ]:
        if candidate.exists():
            odds_path = candidate
            break

    if not odds_path.exists():
        return None

    odds_df = pd.read_csv(odds_path)
    if "implied_probability" not in odds_df.columns:
        return None

    merged = tournament_probs[["team", "prob_winner"]].merge(
        odds_df[["team", "implied_probability"]].rename(columns={"implied_probability": "odds_prob"}),
        on="team",
        how="inner",
    )
    merged["model_prob"] = merged["prob_winner"]
    merged["diff_pct"] = (merged["model_prob"] - merged["odds_prob"]) * 100
    merged = merged.sort_values("model_prob", ascending=False).reset_index(drop=True)

    return merged[["team", "model_prob", "odds_prob", "diff_pct"]]


def compute_surprise_matches(match_preds: pd.DataFrame) -> pd.DataFrame:
    match_preds = match_preds.copy()
    match_preds["surprise_score"] = np.where(
        match_preds["prediction"] == "away_win",
        match_preds["prob_away_win"] - match_preds["prob_home_win"],
        np.where(
            match_preds["prediction"] == "draw",
            match_preds["prob_draw"] - match_preds[["prob_home_win", "prob_away_win"]].max(axis=1),
            0,
        ),
    )
    surprises = match_preds[match_preds["surprise_score"] > 0].sort_values(
        "surprise_score", ascending=False
    ).head(10)
    return surprises[["group", "home_team", "away_team", "prob_home_win", "prob_draw", "prob_away_win", "prediction", "surprise_score"]]


def compute_most_competitive_groups(group_probs: pd.DataFrame) -> pd.DataFrame:
    group_stats = []
    for group in sorted(group_probs["group"].unique()):
        g = group_probs[group_probs["group"] == group]
        advance_spread = g["prob_advance"].max() - g["prob_advance"].min()
        avg_advance = g["prob_advance"].mean()
        most_likely_1st = g.loc[g["prob_1st"].idxmax(), "team"]
        most_likely_1st_pct = g["prob_1st"].max() * 100
        group_stats.append(
            {
                "group": group,
                "advance_spread": advance_spread * 100,
                "avg_advance_pct": avg_advance * 100,
                "most_likely_1st": most_likely_1st,
                "most_likely_1st_pct": most_likely_1st_pct,
                "is_tight": advance_spread < 0.40,
            }
        )

    return pd.DataFrame(group_stats).sort_values("advance_spread").reset_index(drop=True)