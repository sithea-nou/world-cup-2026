import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    CONFEDERATIONS,
    HOST_NATIONS,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_STATE,
)
from src.features.elo import EloRatingSystem
from src.helpers import logger, normalize_team_name


FEATURES_CACHE_VERSION = "7"


def _compute_input_hash(include_live: bool, debug: bool = False) -> str:
    """Compute a hash of the input files that affect match feature engineering."""
    # Only include files whose contents actually affect historical match features.
    # Current rankings/odds affect WC2026 features and are handled separately.
    files = [
        (RAW_DIR / "international_matches" / "results.csv", "match_results"),
        (RAW_DIR / "international_matches" / "shootouts.csv", "shootouts"),
        (RAW_DIR / "fifa_rankings" / "fifa_ranking.csv", "historical_rankings"),
        (Path(__file__).parent.parent.parent / "data" / "external" / "continents.csv", "continents"),
        (RAW_DIR / "squad_quality.csv", "squad_quality"),
    ]
    if include_live:
        files.append((RAW_DIR / "wc2026_results_live.csv", "live_results"))

    hasher = hashlib.sha256()
    hasher.update(FEATURES_CACHE_VERSION.encode())
    hasher.update(str(include_live).encode())

    for f, label in files:
        if f.exists():
            if debug:
                logger.info(f"  Cache input {label}: {f} ({f.stat().st_size} bytes)")
            hasher.update(f.read_bytes())
        else:
            if debug:
                logger.info(f"  Cache input {label}: missing")
            hasher.update(b"missing")

    return hasher.hexdigest()[:16]


def _load_cached_features(include_live: bool) -> pd.DataFrame | None:
    cache_path = PROCESSED_DIR / ".match_features_hash"
    features_path = PROCESSED_DIR / "match_features.parquet"

    if not features_path.exists() or not cache_path.exists():
        return None

    current_hash = _compute_input_hash(include_live, debug=False)
    cached_hash = cache_path.read_text().strip()

    if current_hash != cached_hash:
        logger.info("Feature cache hash mismatch, rebuilding features...")
        if logger.isEnabledFor("DEBUG"):
            _compute_input_hash(include_live, debug=True)
        return None

    logger.info("Loading cached match features (inputs unchanged)...")
    return pd.read_parquet(features_path)


def _save_cached_features_hash(include_live: bool):
    cache_path = PROCESSED_DIR / ".match_features_hash"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(_compute_input_hash(include_live))


def load_all_data(
    include_live: bool = False,
) -> dict[str, pd.DataFrame]:
    logger.info("Loading all data sources...")

    data = {}

    results_path = RAW_DIR / "international_matches" / "results.csv"
    if results_path.exists():
        data["matches"] = pd.read_csv(results_path)
        data["matches"]["date"] = pd.to_datetime(data["matches"]["date"])
        data["matches"]["home_team"] = data["matches"]["home_team"].apply(normalize_team_name)
        data["matches"]["away_team"] = data["matches"]["away_team"].apply(normalize_team_name)
        logger.info(f"  Matches: {len(data['matches'])} rows")
    else:
        logger.warning(f"Match results not found at {results_path}. Run download_kaggle.py first.")

    shootouts_path = RAW_DIR / "international_matches" / "shootouts.csv"
    if shootouts_path.exists():
        data["shootouts"] = pd.read_csv(shootouts_path)
        data["shootouts"]["date"] = pd.to_datetime(data["shootouts"]["date"])
        data["shootouts"]["home_team"] = data["shootouts"]["home_team"].apply(normalize_team_name)
        data["shootouts"]["away_team"] = data["shootouts"]["away_team"].apply(normalize_team_name)
        logger.info(f"  Shootouts: {len(data['shootouts'])} rows")

    rankings_path = RAW_DIR / "fifa_rankings_merged.csv"
    if not rankings_path.exists():
        rankings_path = RAW_DIR / "fifa_rankings" / "fifa_ranking.csv"
    if rankings_path.exists():
        data["rankings"] = pd.read_csv(rankings_path)
        data["rankings"]["rank_date"] = pd.to_datetime(data["rankings"]["rank_date"], format="mixed")
        if "country_full" in data["rankings"].columns:
            data["rankings"]["country"] = data["rankings"]["country_full"].apply(normalize_team_name)
        elif "country" in data["rankings"].columns:
            data["rankings"]["country"] = data["rankings"]["country"].apply(normalize_team_name)
        logger.info(f"  Rankings: {len(data['rankings'])} rows")
    else:
        logger.warning("FIFA rankings not found. Run scrape_fifa_rankings.py first.")

    current_rankings_path = RAW_DIR / "fifa_rankings_current.csv"
    if current_rankings_path.exists():
        data["current_rankings"] = pd.read_csv(current_rankings_path)
        data["current_rankings"]["country"] = data["current_rankings"]["country"].apply(normalize_team_name)
        logger.info(f"  Current rankings: {len(data['current_rankings'])} rows")

    continents_path = Path(__file__).parent.parent.parent / "data" / "external" / "continents.csv"
    if continents_path.exists():
        data["continents"] = pd.read_csv(continents_path)
        data["continents"]["country"] = data["continents"]["country"].apply(normalize_team_name)
        logger.info(f"  Continents: {len(data['continents'])} rows")

    wc_history_path = RAW_DIR / "historical_world_cups.csv"
    if wc_history_path.exists():
        data["wc_history"] = pd.read_csv(wc_history_path)
        if "date" in data["wc_history"].columns:
            data["wc_history"]["date"] = pd.to_datetime(data["wc_history"]["date"])
        if "home_team" in data["wc_history"].columns:
            data["wc_history"]["home_team"] = data["wc_history"]["home_team"].apply(normalize_team_name)
        if "away_team" in data["wc_history"].columns:
            data["wc_history"]["away_team"] = data["wc_history"]["away_team"].apply(normalize_team_name)
        logger.info(f"  WC History: {len(data['wc_history'])} rows")

    if include_live:
        live_path = RAW_DIR / "wc2026_results_live.csv"
        if live_path.exists():
            data["live_results"] = pd.read_csv(live_path)
            if "date" in data["live_results"].columns:
                data["live_results"]["date"] = pd.to_datetime(data["live_results"]["date"])
            logger.info(f"  Live results: {len(data['live_results'])} rows")

    return data


def load_squad_quality() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    squad_path = RAW_DIR / "squad_quality.csv"
    if not squad_path.exists():
        logger.warning(f"Squad quality data not found at {squad_path}")
        return pd.DataFrame(), {}

    df = pd.read_csv(squad_path)
    df["team"] = df["team"].apply(normalize_team_name)
    logger.info(f"  Squad quality: {len(df)} teams")

    confed_avgs = {}
    for confed in df.get("confederation", pd.Series()).unique() if "confederation" in df.columns else []:
        subset = df[df["confederation"] == confed]
        avgs = {}
        for col in ["squad_market_value_m", "avg_player_value_m", "top_player_value_m"]:
            if col in subset.columns:
                avgs[col] = subset[col].mean()
        if avgs:
            confed_avgs[confed] = avgs

    if not confed_avgs and "squad_market_value_m" in df.columns:
        confed_avgs["__global__"] = {
            "squad_market_value_m": df["squad_market_value_m"].mean(),
            "avg_player_value_m": df.get("avg_player_value_m", pd.Series([0])).mean(),
            "top_player_value_m": df.get("top_player_value_m", pd.Series([0])).mean(),
        }

    return df, confed_avgs


def get_team_squad_value(team: str, squad_df: pd.DataFrame, confed_avgs: dict, confederation: str = "") -> dict:
    if squad_df.empty:
        return {
            "squad_market_value_m": np.nan,
            "avg_player_value_m": np.nan,
            "top_player_value_m": np.nan,
        }

    match = squad_df[squad_df["team"] == team]
    if not match.empty:
        row = match.iloc[0]
        return {
            "squad_market_value_m": row.get("squad_market_value_m", np.nan),
            "avg_player_value_m": row.get("avg_player_value_m", np.nan),
            "top_player_value_m": row.get("top_player_value_m", np.nan),
        }

    fallback = confed_avgs.get(confederation, confed_avgs.get("__global__", {}))
    return {
        "squad_market_value_m": fallback.get("squad_market_value_m", np.nan),
        "avg_player_value_m": fallback.get("avg_player_value_m", np.nan),
        "top_player_value_m": fallback.get("top_player_value_m", np.nan),
    }


def _get_fifa_ranking(team: str, date: pd.Timestamp, rankings_df: pd.DataFrame) -> dict:
    if rankings_df is None or rankings_df.empty:
        return {"fifa_rank": np.nan, "fifa_points": np.nan}

    country_col = "country" if "country" in rankings_df.columns else "country_full"
    team_rankings = rankings_df[rankings_df[country_col] == team]

    if "rank_date" in team_rankings.columns:
        team_rankings = team_rankings[team_rankings["rank_date"] <= date]

    if team_rankings.empty:
        return {"fifa_rank": np.nan, "fifa_points": np.nan}

    if "rank_date" in team_rankings.columns:
        latest = team_rankings.sort_values("rank_date").iloc[-1]
    else:
        latest = team_rankings.iloc[-1]

    rank = latest.get("rank", np.nan)
    points = latest.get("total_points", np.nan)

    return {
        "fifa_rank": float(rank) if not pd.isna(rank) else np.nan,
        "fifa_points": float(points) if not pd.isna(points) else np.nan,
    }


def _get_confederation(team: str, continents_df: pd.DataFrame | None = None) -> str:
    if continents_df is not None and not continents_df.empty:
        match = continents_df[continents_df["country"] == team]
        if not match.empty:
            return match.iloc[0]["confederation"]
    return CONFEDERATIONS.get(team, "Unknown")


def _compute_form_features(
    team: str, date: pd.Timestamp, matches_df: pd.DataFrame, n: int = 10
) -> dict:
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team))
        & (matches_df["date"] < date)
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return {
            f"form_last{n}_win_rate": np.nan,
            f"form_last{n}_draw_rate": np.nan,
            f"form_last{n}_loss_rate": np.nan,
            f"form_last{n}_goals_scored_avg": np.nan,
            f"form_last{n}_goals_conceded_avg": np.nan,
            f"form_last{n}_goal_diff_avg": np.nan,
            f"form_last{n}_clean_sheet_rate": np.nan,
        }

    wins = 0
    draws = 0
    losses = 0
    goals_scored = 0
    goals_conceded = 0
    clean_sheets = 0

    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            goals_scored += row.get("home_score", 0) if not pd.isna(row.get("home_score")) else 0
            goals_conceded += row.get("away_score", 0) if not pd.isna(row.get("away_score")) else 0
            if row.get("home_score", 0) is not np.nan and row.get("away_score", 0) is not np.nan:
                if row["home_score"] > row["away_score"]:
                    wins += 1
                elif row["home_score"] == row["away_score"]:
                    draws += 1
                else:
                    losses += 1
                if row["away_score"] == 0:
                    clean_sheets += 1
        else:
            goals_scored += row.get("away_score", 0) if not pd.isna(row.get("away_score")) else 0
            goals_conceded += row.get("home_score", 0) if not pd.isna(row.get("home_score")) else 0
            if row.get("home_score", 0) is not np.nan and row.get("away_score", 0) is not np.nan:
                if row["away_score"] > row["home_score"]:
                    wins += 1
                elif row["away_score"] == row["home_score"]:
                    draws += 1
                else:
                    losses += 1
                if row["home_score"] == 0:
                    clean_sheets += 1

    num = len(team_matches)
    result = {
        f"form_last{n}_win_rate": wins / num,
        f"form_last{n}_draw_rate": draws / num,
        f"form_last{n}_loss_rate": losses / num,
        f"form_last{n}_goals_scored_avg": goals_scored / num,
        f"form_last{n}_goals_conceded_avg": goals_conceded / num,
        f"form_last{n}_goal_diff_avg": (goals_scored - goals_conceded) / num,
        f"form_last{n}_clean_sheet_rate": clean_sheets / num,
    }

    if num >= 3:
        weights = np.exp(-0.3 * np.arange(num))
        weights /= weights.sum()
        w_wins = 0.0
        w_draws = 0.0
        w_losses = 0.0
        w_goals_scored = 0.0
        w_goals_conceded = 0.0
        for i, (_, row) in enumerate(team_matches.iterrows()):
            w = weights[i]
            if row["home_team"] == team:
                gs = row.get("home_score", 0) if not pd.isna(row.get("home_score")) else 0
                gc = row.get("away_score", 0) if not pd.isna(row.get("away_score")) else 0
                if not pd.isna(row.get("home_score")) and not pd.isna(row.get("away_score")):
                    if row["home_score"] > row["away_score"]:
                        w_wins += w
                    elif row["home_score"] == row["away_score"]:
                        w_draws += w
                    else:
                        w_losses += w
            else:
                gs = row.get("away_score", 0) if not pd.isna(row.get("away_score")) else 0
                gc = row.get("home_score", 0) if not pd.isna(row.get("home_score")) else 0
                if not pd.isna(row.get("home_score")) and not pd.isna(row.get("away_score")):
                    if row["away_score"] > row["home_score"]:
                        w_wins += w
                    elif row["away_score"] == row["home_score"]:
                        w_draws += w
                    else:
                        w_losses += w
            w_goals_scored += w * gs
            w_goals_conceded += w * gc
        w_total = w_wins + w_draws + w_losses if (w_wins + w_draws + w_losses) > 0 else 1.0
        result[f"form_last{n}_ewm_win_rate"] = w_wins / w_total
        result[f"form_last{n}_ewm_draw_rate"] = w_draws / w_total
        result[f"form_last{n}_ewm_loss_rate"] = w_losses / w_total
        result[f"form_last{n}_ewm_goals_scored_avg"] = w_goals_scored
        result[f"form_last{n}_ewm_goals_conceded_avg"] = w_goals_conceded
    else:
        result[f"form_last{n}_ewm_win_rate"] = result[f"form_last{n}_win_rate"]
        result[f"form_last{n}_ewm_draw_rate"] = result[f"form_last{n}_draw_rate"]
        result[f"form_last{n}_ewm_loss_rate"] = result[f"form_last{n}_loss_rate"]
        result[f"form_last{n}_ewm_goals_scored_avg"] = result[f"form_last{n}_goals_scored_avg"]
        result[f"form_last{n}_ewm_goals_conceded_avg"] = result[f"form_last{n}_goals_conceded_avg"]

    return result


def _compute_sos_features(
    team: str, date: pd.Timestamp, matches_df: pd.DataFrame, elo_system, n: int = 10
) -> dict:
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team))
        & (matches_df["date"] < date)
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return {
            "sos_avg_opp_elo": np.nan,
            "sos_avg_opp_fifa_rank": np.nan,
        }

    opp_elos = []
    for _, row in team_matches.iterrows():
        opp = row["away_team"] if row["home_team"] == team else row["home_team"]
        opp_rating = elo_system.get_team_rating(opp)
        opp_elos.append(opp_rating)

    return {
        "sos_avg_opp_elo": np.mean(opp_elos) if opp_elos else np.nan,
        "sos_avg_opp_fifa_rank": np.nan,
    }


def _compute_h2h_features(
    home: str, away: str, date: pd.Timestamp, matches_df: pd.DataFrame, n: int = 5
) -> dict:
    h2h = matches_df[
        (
            ((matches_df["home_team"] == home) & (matches_df["away_team"] == away))
            | ((matches_df["home_team"] == away) & (matches_df["away_team"] == home))
        )
        & (matches_df["date"] < date)
    ].sort_values("date", ascending=False).head(n)

    if h2h.empty:
        return {
            "h2h_home_wins": np.nan,
            "h2h_draws": np.nan,
            "h2h_away_wins": np.nan,
            "h2h_home_goals_avg": np.nan,
            "h2h_away_goals_avg": np.nan,
        }

    home_wins = 0
    draws = 0
    away_wins = 0
    home_goals = 0
    away_goals = 0

    for _, row in h2h.iterrows():
        hs = row.get("home_score", 0)
        as_ = row.get("away_score", 0)
        if pd.isna(hs) or pd.isna(as_):
            continue

        if row["home_team"] == home:
            home_goals += hs
            away_goals += as_
            if hs > as_:
                home_wins += 1
            elif hs == as_:
                draws += 1
            else:
                away_wins += 1
        else:
            home_goals += as_
            away_goals += hs
            if as_ > hs:
                home_wins += 1
            elif hs == as_:
                draws += 1
            else:
                away_wins += 1

    num = len(h2h)
    return {
        "h2h_home_wins": home_wins,
        "h2h_draws": draws,
        "h2h_away_wins": away_wins,
        "h2h_draw_rate": draws / num,
        "h2h_home_goals_avg": home_goals / num,
        "h2h_away_goals_avg": away_goals / num,
    }


def _compute_form_features_cached(
    cache: dict,
    team: str,
    date: pd.Timestamp,
    matches_df: pd.DataFrame,
    n: int = 10,
) -> dict:
    key = (team, date, n)
    if key in cache:
        return cache[key]

    result = _compute_form_features(team, date, matches_df, n)
    cache[key] = result
    return result


def _compute_h2h_features_cached(
    cache: dict,
    home: str,
    away: str,
    date: pd.Timestamp,
    matches_df: pd.DataFrame,
    n: int = 5,
) -> dict:
    key = (home, away, date, n)
    if key in cache:
        return cache[key]

    result = _compute_h2h_features(home, away, date, matches_df, n)
    cache[key] = result
    return result


def _precompute_fifa_rankings(
    rankings_df: pd.DataFrame, matches_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Precompute nearest FIFA ranking per team per match date using merge_asof."""
    if rankings_df is None or rankings_df.empty:
        return {}

    country_col = "country" if "country" in rankings_df.columns else "country_full"
    df = rankings_df.copy()
    df[country_col] = df[country_col].apply(normalize_team_name)
    df = df.sort_values("rank_date")

    teams = set(matches_df["home_team"]).union(set(matches_df["away_team"]))
    rankings_by_team = {}
    for team in teams:
        team_df = df[df[country_col] == team][["rank_date", "rank", "total_points"]].copy()
        if team_df.empty:
            continue
        rankings_by_team[team] = team_df
    return rankings_by_team


def _get_fifa_ranking_fast(
    team: str,
    date: pd.Timestamp,
    rankings_by_team: dict[str, pd.DataFrame],
) -> dict:
    team_df = rankings_by_team.get(team)
    if team_df is None or team_df.empty:
        return {"fifa_rank": np.nan, "fifa_points": np.nan}

    past = team_df[team_df["rank_date"] <= date]
    if past.empty:
        return {"fifa_rank": np.nan, "fifa_points": np.nan}

    latest = past.iloc[-1]
    rank = latest.get("rank", np.nan)
    points = latest.get("total_points", np.nan)
    return {
        "fifa_rank": float(rank) if not pd.isna(rank) else np.nan,
        "fifa_points": float(points) if not pd.isna(points) else np.nan,
    }


def build_match_features(
    matches_df: pd.DataFrame | None = None,
    elo_system: EloRatingSystem | None = None,
    include_live: bool = False,
) -> pd.DataFrame:
    logger.info("Building match features...")

    cached = _load_cached_features(include_live)
    if cached is not None:
        logger.info(f"Using cached match features ({len(cached)} rows)")
        return cached

    data = load_all_data(include_live=include_live)
    if matches_df is None:
        matches_df = data.get("matches")
        if matches_df is None:
            raise ValueError("No match data available. Run scraping first.")

    if elo_system is None:
        elo_system = EloRatingSystem()
        elo_system.compute_elo_ratings(matches_df)

    rankings_df = data.get("rankings", pd.DataFrame())
    continents_df = data.get("continents", pd.DataFrame())

    rankings_by_team = _precompute_fifa_rankings(rankings_df, matches_df)
    form_cache: dict = {}
    h2h_cache: dict = {}

    squad_df, confed_avgs = load_squad_quality()
    if not squad_df.empty and "confederation" not in squad_df.columns:
        for idx, row in squad_df.iterrows():
            confed = _get_confederation(row["team"], continents_df)
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

    wc_history_df = data.get("wc_history", pd.DataFrame())
    knockout_keys = set()
    if not wc_history_df.empty and "round" in wc_history_df.columns:
        ko = wc_history_df[wc_history_df["round"].str.lower().str.contains("round|quarter|semi|final|third", na=False)]
        for _, r in ko.iterrows():
            h = r.get("home_team", "")
            a = r.get("away_team", "")
            if h and a:
                knockout_keys.add((normalize_team_name(str(h)), normalize_team_name(str(a))))

    wc_knockout_dates = {}
    if "tournament" in matches_df.columns:
        wc_matches_mask = matches_df["tournament"].str.contains("World Cup", case=False, na=False) & ~matches_df["tournament"].str.contains("qualification|qualifying", case=False, na=False)
        wc_matches = matches_df[wc_matches_mask]
        for year, group in wc_matches.groupby(matches_df.loc[wc_matches_mask, "date"].dt.year):
            dates = group["date"].sort_values()
            if len(dates) < 16:
                continue
            start = dates.min()
            group_end = start + pd.Timedelta(days=15)
            wc_knockout_dates[year] = group_end

    features_list = []

    valid_matches = matches_df[
        matches_df["home_score"].notna() & matches_df["away_score"].notna()
    ].copy()
    valid_matches = valid_matches.sort_values("date").reset_index(drop=True)

    for idx, row in tqdm(valid_matches.iterrows(), total=len(valid_matches), desc="Building features"):
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        home_elo = elo_system.get_team_rating(home)
        away_elo = elo_system.get_team_rating(away)

        neutral = row.get("neutral", True)
        if isinstance(neutral, str):
            neutral = neutral.lower() == "true"
        neutral = bool(neutral)

        probs = elo_system.predict_match_probability(home_elo, away_elo, neutral=neutral)

        home_ranking = _get_fifa_ranking_fast(home, date, rankings_by_team)
        away_ranking = _get_fifa_ranking_fast(away, date, rankings_by_team)

        home_form_10 = _compute_form_features_cached(form_cache, home, date, matches_df, n=10)
        home_form_5 = _compute_form_features_cached(form_cache, home, date, matches_df, n=5)
        away_form_10 = _compute_form_features_cached(form_cache, away, date, matches_df, n=10)
        away_form_5 = _compute_form_features_cached(form_cache, away, date, matches_df, n=5)

        h2h = _compute_h2h_features_cached(h2h_cache, home, away, date, matches_df, n=5)

        home_sos = _compute_sos_features(home, date, matches_df, elo_system, n=10)
        away_sos = _compute_sos_features(away, date, matches_df, elo_system, n=10)

        home_confed = _get_confederation(home, continents_df)
        away_confed = _get_confederation(away, continents_df)
        same_confed = 1 if home_confed == away_confed else 0

        home_squad = get_team_squad_value(home, squad_df, confed_avgs, home_confed)
        away_squad = get_team_squad_value(away, squad_df, confed_avgs, away_confed)

        tournament = str(row.get("tournament", "Friendly"))
        is_world_cup = 1 if "World Cup" in tournament and "qualification" not in tournament.lower() else 0
        is_qualifier = 1 if "qualification" in tournament.lower() or "qualifying" in tournament.lower() else 0
        is_friendly = 1 if "Friendly" in tournament or "friendly" in tournament else 0

        home_advantage = 1.0 if not neutral else 0.5
        is_host = 1 if home in HOST_NATIONS else 0

        home_score = row["home_score"]
        away_score = row["away_score"]
        if home_score > away_score:
            outcome = 1
        elif home_score < away_score:
            outcome = -1
        else:
            outcome = 0

        goal_diff = home_score - away_score

        features = {
            "date": date,
            "home_team": home,
            "away_team": away,
            "tournament": tournament,
            "neutral": int(neutral),
            "home_advantage": home_advantage,
            "is_host_nation": is_host,
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_delta": home_elo - away_elo,
            "elo_abs_delta": abs(home_elo - away_elo),
            "elo_home_win_prob": probs["home_win"],
            "elo_draw_prob": probs["draw"],
            "elo_away_win_prob": probs["away_win"],
            "fifa_rank_home": home_ranking["fifa_rank"],
            "fifa_rank_away": away_ranking["fifa_rank"],
            "fifa_rank_delta": home_ranking["fifa_rank"] - away_ranking["fifa_rank"],
            "fifa_rank_abs_delta": abs(home_ranking["fifa_rank"] - away_ranking["fifa_rank"]),
            "fifa_points_home": home_ranking["fifa_points"],
            "fifa_points_away": away_ranking["fifa_points"],
            "fifa_points_delta": home_ranking["fifa_points"] - away_ranking["fifa_points"],
            "fifa_points_abs_delta": abs(home_ranking["fifa_points"] - away_ranking["fifa_points"]),
            "same_confederation": same_confed,
            "is_world_cup": is_world_cup,
            "is_qualifier": is_qualifier,
            "is_friendly": is_friendly,
            "is_knockout": int(is_world_cup and ((home, away) in knockout_keys or (away, home) in knockout_keys or "knockout" in tournament.lower() or "round" in tournament.lower() or "final" in tournament.lower() or (date.year in wc_knockout_dates and date >= wc_knockout_dates[date.year]))),
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
            "outcome": outcome,
            "goal_diff": goal_diff,
            "home_score": home_score,
            "away_score": away_score,
        }

        features_list.append(features)

    df = pd.DataFrame(features_list)

    # Derived interaction / draw-calibration features
    df["elo_delta_x_home_advantage"] = df["elo_delta"].astype(float) * df["home_advantage"].astype(float)
    df["fifa_rank_delta_x_same_confed"] = df["fifa_rank_delta"].astype(float) * df["same_confederation"].astype(float)
    home_draw = df.get("home_form_last10_draw_rate", pd.Series(np.nan, index=df.index))
    away_draw = df.get("away_form_last10_draw_rate", pd.Series(np.nan, index=df.index))
    df["combined_draw_prob"] = (df["elo_draw_prob"].astype(float) + 0.5 * (home_draw.fillna(0) + away_draw.fillna(0))) / 2
    df["combined_draw_prob"] = df["combined_draw_prob"].fillna(df["elo_draw_prob"].astype(float))

    df["elo_close"] = (df["elo_abs_delta"] < 100).astype(int)
    df["draw_tendency"] = df["combined_draw_prob"] * (1 + 0.5 * df["elo_close"].astype(float))
    df["fifa_close"] = (df["fifa_rank_abs_delta"] < 20).astype(int)

    draw_rate_map = {
        "friendly": 0.26,
        "qualifier": 0.24,
        "world_cup_group": 0.23,
        "world_cup_knockout": 0.15,
        "other": 0.25,
    }
    def _tournament_draw_rate(row):
        if row.get("is_world_cup", 0) == 1:
            return draw_rate_map["world_cup_knockout"] if row.get("is_knockout", 0) == 1 else draw_rate_map["world_cup_group"]
        elif row.get("is_qualifier", 0) == 1:
            return draw_rate_map["qualifier"]
        elif row.get("is_friendly", 0) == 1:
            return draw_rate_map["friendly"]
        return draw_rate_map["other"]

    df["tournament_draw_rate"] = df.apply(_tournament_draw_rate, axis=1)

    out_path = PROCESSED_DIR / "match_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    _save_cached_features_hash(include_live)
    logger.info(f"Saved {len(df)} match features to {out_path}")

    return df


if __name__ == "__main__":
    df = build_match_features()
    print(f"Features: {df.shape}")
    print(df.head())
    print(f"\nOutcome distribution:")
    print(df["outcome"].value_counts())