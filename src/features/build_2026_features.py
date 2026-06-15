from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.features.elo import EloRatingSystem
from src.helpers import logger, normalize_team_name


def build_wc2026_features(
    groups_df: pd.DataFrame | None = None,
    fixtures_df: pd.DataFrame | None = None,
    elo_system: EloRatingSystem | None = None,
    rankings_df: pd.DataFrame | None = None,
    odds_df: pd.DataFrame | None = None,
    include_live: bool = True,
) -> pd.DataFrame:
    logger.info("Building WC 2026 match features...")

    if groups_df is None:
        groups_path = RAW_DIR / "wc2026_groups.csv"
        if groups_path.exists():
            groups_df = pd.read_csv(groups_path)
        else:
            logger.warning("WC 2026 groups not found. Run scrape_world_cup_2026.py first.")
            return pd.DataFrame()

    if fixtures_df is None:
        fixtures_path = RAW_DIR / "wc2026_fixtures.csv"
        if fixtures_path.exists():
            fixtures_df = pd.read_csv(fixtures_path)
        else:
            logger.warning("WC 2026 fixtures not found. Run scrape_world_cup_2026.py first.")
            return pd.DataFrame()

    if elo_system is None:
        from src.features.elo import EloRatingSystem

        elo_path = PROCESSED_DIR / "elo_ratings_current.parquet"
        if elo_path.exists():
            elo_ratings = pd.read_parquet(elo_path)
            if len(elo_ratings) > 10:
                elo_system = EloRatingSystem()
                for _, row in elo_ratings.iterrows():
                    elo_system.ratings[row["team"]] = row["elo"]
            else:
                logger.warning(f"Elo cache has only {len(elo_ratings)} teams, recomputing from match history")

        if elo_system is None:
            logger.info("Computing fresh Elo ratings from match history...")
            raw_matches = pd.read_csv(RAW_DIR / "international_matches" / "results.csv")
            raw_matches["date"] = pd.to_datetime(raw_matches["date"])
            raw_matches["home_team"] = raw_matches["home_team"].apply(normalize_team_name)
            raw_matches["away_team"] = raw_matches["away_team"].apply(normalize_team_name)
            elo_system = EloRatingSystem()
            elo_system.compute_elo_ratings(raw_matches)

    if rankings_df is None:
        current_path = RAW_DIR / "fifa_rankings_current.csv"
        if current_path.exists():
            rankings_df = pd.read_csv(current_path)
            rankings_df["country"] = rankings_df["country"].apply(normalize_team_name)
        else:
            rankings_df = pd.DataFrame()

    if odds_df is None:
        odds_path = RAW_DIR / "odds_outright.csv"
        if odds_path.exists():
            odds_df = pd.read_csv(odds_path)
        else:
            odds_df = pd.DataFrame()

    match_odds_path = RAW_DIR / "odds_match.csv"
    match_odds_df = pd.DataFrame()
    if match_odds_path.exists():
        match_odds_df = pd.read_csv(match_odds_path)

    from src.features.build_features import (
        _get_confederation,
        _get_fifa_ranking,
        get_team_squad_value,
        load_all_data,
        load_squad_quality,
    )

    data = load_all_data(include_live=include_live)
    matches_df = data.get("matches", pd.DataFrame())
    continents_df = data.get("continents", pd.DataFrame())

    squad_df, confed_avgs = load_squad_quality()
    if not squad_df.empty and "confederation" not in squad_df.columns:
        from src.features.build_features import _get_confederation as _get_confed
        for idx, row in squad_df.iterrows():
            confed = _get_confed(row["team"], continents_df)
            squad_df.at[idx, "confederation"] = confed
        confed_avgs = {}
        for confed in squad_df["confederation"].unique():
            subset = squad_df[squad_df["confederation"] == confed]
            avgs = {}
            for col in ["squad_market_value_m", "avg_player_value_m", "top_player_value_m"]:
                if col in subset.columns:
                    avgs[col] = subset[col].mean()
            if avgs:
                confed_avgs[confed] = avgs

    group_map = {}
    if groups_df is not None and not groups_df.empty:
        for _, row in groups_df.iterrows():
            group_map[normalize_team_name(str(row["team"]))] = row.get("group", "Unknown")

    wc_start_date = pd.Timestamp("2026-06-11")

    features_list = []

    for _, row in fixtures_df.iterrows():
        home = normalize_team_name(str(row.get("home_team", "")))
        away = normalize_team_name(str(row.get("away_team", "")))

        if not home or not away:
            continue

        home_elo = elo_system.get_team_rating(home)
        away_elo = elo_system.get_team_rating(away)

        probs = elo_system.predict_match_probability(home_elo, away_elo, neutral=True)

        home_ranking = _get_fifa_ranking(home, wc_start_date, rankings_df)
        away_ranking = _get_fifa_ranking(away, wc_start_date, rankings_df)

        home_confed = _get_confederation(home, continents_df)
        away_confed = _get_confederation(away, continents_df)
        same_confed = 1 if home_confed == away_confed else 0

        home_form_10 = {}
        home_form_5 = {}
        away_form_10 = {}
        away_form_5 = {}

        if not matches_df.empty:
            from src.features.build_features import _compute_form_features, _compute_h2h_features, _compute_sos_features

            home_form_10 = _compute_form_features(home, wc_start_date, matches_df, n=10)
            home_form_5 = _compute_form_features(home, wc_start_date, matches_df, n=5)
            away_form_10 = _compute_form_features(away, wc_start_date, matches_df, n=10)
            away_form_5 = _compute_form_features(away, wc_start_date, matches_df, n=5)

        h2h = {}
        if not matches_df.empty:
            from src.features.build_features import _compute_h2h_features

            h2h = _compute_h2h_features(home, away, wc_start_date, matches_df, n=5)

        home_sos = _compute_sos_features(home, wc_start_date, matches_df, elo_system, n=10) if not matches_df.empty else {"sos_avg_opp_elo": np.nan}
        away_sos = _compute_sos_features(away, wc_start_date, matches_df, elo_system, n=10) if not matches_df.empty else {"sos_avg_opp_elo": np.nan}

        home_squad = get_team_squad_value(home, squad_df, confed_avgs, home_confed)
        away_squad = get_team_squad_value(away, squad_df, confed_avgs, away_confed)

        home_implied_prob = np.nan
        away_implied_prob = np.nan
        draw_implied_prob = np.nan

        if not match_odds_df.empty:
            match_odd = match_odds_df[
                (match_odds_df["home_team"] == home) & (match_odds_df["away_team"] == away)
            ]
            if match_odd.empty:
                match_odd = match_odds_df[
                    (match_odds_df["home_team"] == away) & (match_odds_df["away_team"] == home)
                ]
            if not match_odd.empty:
                row_odd = match_odd.iloc[0]
                home_implied_prob = row_odd.get("home_implied_prob", np.nan)
                draw_implied_prob = row_odd.get("draw_implied_prob", np.nan)
                away_implied_prob = row_odd.get("away_implied_prob", np.nan)

        is_host = 1 if home in ["United States", "Canada", "Mexico"] else 0
        is_away_host = 1 if away in ["United States", "Canada", "Mexico"] else 0
        is_neutral = 0 if (is_host or is_away_host) else 1
        home_adv = 1.0 if is_host else (0.5 if is_neutral else 0.3 if is_away_host else 0.5)

        features = {
            "date": row.get("date", wc_start_date),
            "home_team": home,
            "away_team": away,
            "group": row.get("group", group_map.get(home, "Unknown")),
            "match_number": row.get("match_number", 0),
            "tournament": "FIFA World Cup",
            "neutral": is_neutral,
            "home_advantage": home_adv,
            "is_host_nation": is_host,
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_delta": home_elo - away_elo,
            "elo_home_win_prob": probs["home_win"],
            "elo_draw_prob": probs["draw"],
            "elo_away_win_prob": probs["away_win"],
            "fifa_rank_home": home_ranking.get("fifa_rank", np.nan),
            "fifa_rank_away": away_ranking.get("fifa_rank", np.nan),
            "fifa_rank_delta": home_ranking.get("fifa_rank", np.nan) - away_ranking.get("fifa_rank", np.nan),
            "fifa_rank_abs_delta": abs(home_ranking.get("fifa_rank", np.nan) - away_ranking.get("fifa_rank", np.nan)),
            "fifa_points_home": home_ranking.get("fifa_points", np.nan),
            "fifa_points_away": away_ranking.get("fifa_points", np.nan),
            "fifa_points_delta": home_ranking.get("fifa_points", np.nan) - away_ranking.get("fifa_points", np.nan),
            "fifa_points_abs_delta": abs(home_ranking.get("fifa_points", np.nan) - away_ranking.get("fifa_points", np.nan)),
            "same_confederation": same_confed,
            "is_world_cup": 1,
            "is_qualifier": 0,
            "is_friendly": 0,
            "is_knockout": 0,
            "odds_home_implied_prob": home_implied_prob,
            "odds_draw_implied_prob": draw_implied_prob,
            "odds_away_implied_prob": away_implied_prob,
            **{f"home_{k}": v for k, v in home_form_10.items()},
            **{f"home_{k}": v for k, v in home_form_5.items()},
            **{f"away_{k}": v for k, v in away_form_10.items()},
            **{f"away_{k}": v for k, v in away_form_5.items()},
            **h2h,
            "home_sos_avg_opp_elo": home_sos["sos_avg_opp_elo"],
            "away_sos_avg_opp_elo": away_sos["sos_avg_opp_elo"],
            "home_squad_value_m": home_squad["squad_market_value_m"],
            "away_squad_value_m": away_squad["squad_market_value_m"],
            "squad_value_delta": home_squad["squad_market_value_m"] - away_squad["squad_market_value_m"],
            "squad_value_abs_delta": abs(home_squad["squad_market_value_m"] - away_squad["squad_market_value_m"]),
            "home_avg_player_value_m": home_squad["avg_player_value_m"],
            "away_avg_player_value_m": away_squad["avg_player_value_m"],
            "home_top_player_value_m": home_squad["top_player_value_m"],
            "away_top_player_value_m": away_squad["top_player_value_m"],
        }

        features_list.append(features)

    df = pd.DataFrame(features_list)

    df["elo_delta_x_home_advantage"] = df["elo_delta"] * df["home_advantage"]
    df["fifa_rank_delta_x_same_confed"] = df["fifa_rank_delta"] * df["same_confederation"]
    df["elo_abs_delta"] = abs(df["elo_delta"])
    df["combined_draw_prob"] = (df["elo_draw_prob"] + 0.5 * (df["home_form_last10_draw_rate"].fillna(0) + df["away_form_last10_draw_rate"].fillna(0))) / 2
    df["combined_draw_prob"] = df["combined_draw_prob"].fillna(df["elo_draw_prob"])

    df["elo_close"] = (df["elo_abs_delta"] < 100).astype(int)
    df["draw_tendency"] = df["combined_draw_prob"] * (1 + 0.5 * df["elo_close"].astype(float))
    df["fifa_close"] = (df["fifa_rank_abs_delta"] < 20).astype(int)
    df["tournament_draw_rate"] = 0.23


    out_path = PROCESSED_DIR / "wc2026_match_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} WC 2026 match features to {out_path}")

    return df


if __name__ == "__main__":
    df = build_wc2026_features()
    print(f"WC 2026 features: {df.shape}")
    print(df.head())