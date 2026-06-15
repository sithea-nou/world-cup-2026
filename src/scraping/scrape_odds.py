import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import ODDS_API_BASE, ODDS_API_KEY, RAW_DIR
from src.helpers import logger, normalize_team_name


def odds_to_probability(american_odds: int) -> float:
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def fetch_outright_odds(api_key: str | None = None) -> pd.DataFrame:
    key = api_key or ODDS_API_KEY
    if not key:
        logger.warning(
            "No ODDS_API_KEY set. Set environment variable ODDS_API_KEY "
            "or pass api_key parameter. Get one free at https://the-odds-api.com/"
        )
        logger.info("Falling back to implied outright odds from FIFA ranking points...")
        return _fallback_outright_odds()

    logger.info("Fetching outright World Cup 2026 odds from the-odds-api.com...")

    url = f"{ODDS_API_BASE}/sports/football/world_cup/odds/"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "outright",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch odds from API: {e}")
        return _fallback_outright_odds()

    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    records = []
    bookmakers = data.get("bookmakers", []) if isinstance(data, dict) else []

    for bm in bookmakers:
        bm_key = bm.get("key", "unknown")
        bm_title = bm.get("title", "Unknown")
        for market in bm.get("markets", []):
            if market.get("key") != "outright":
                continue
            for outcome in market.get("outcomes", []):
                team = outcome.get("name", "")
                odds_val = outcome.get("price", 0)
                try:
                    odds_int = int(odds_val)
                except (ValueError, TypeError):
                    odds_int = 0

                records.append(
                    {
                        "team": team,
                        "american_odds": odds_int,
                        "implied_probability": odds_to_probability(odds_int) if odds_int != 0 else None,
                        "bookmaker": bm_title,
                        "bookmaker_key": bm_key,
                        "last_update": market.get("last_update", ""),
                    }
                )

    df = pd.DataFrame(records)

    json_path = RAW_DIR / "odds_outright.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    if not df.empty:
        df.to_csv(RAW_DIR / "odds_outright.csv", index=False)
        logger.info(f"Saved {len(df)} outright odds to {RAW_DIR / 'odds_outright.csv'}")
        _aggregate_outright_odds(df)
    else:
        logger.warning("No outright odds data found from API, using fallback")
        df = _fallback_outright_odds()

    return df


def _fallback_outright_odds() -> pd.DataFrame:
    """Derive implied outright probabilities from current FIFA ranking points."""
    rankings_path = RAW_DIR / "fifa_rankings_current.csv"
    if not rankings_path.exists():
        logger.warning("No current FIFA rankings available for fallback outright odds")
        return pd.DataFrame()

    rankings = pd.read_csv(rankings_path)
    if "country" not in rankings.columns or "total_points" not in rankings.columns:
        logger.warning("Current FIFA rankings have unexpected columns")
        return pd.DataFrame()

    rankings["country"] = rankings["country"].apply(normalize_team_name)
    points = rankings["total_points"].astype(float).reset_index(drop=True)
    # Softmax-style probability using ranking points
    exp_points = np.exp((points - points.max()) / 200.0)
    probs = exp_points / exp_points.sum()

    records = []
    for idx, row in rankings.iterrows():
        prob = float(probs.iloc[idx])
        # Convert probability to approximately equivalent American odds
        if prob > 0.5:
            american = int(round(-100 * prob / (1 - prob)))
        else:
            american = int(round(100 * (1 - prob) / prob))

        records.append(
            {
                "team": row["country"],
                "american_odds": american,
                "implied_probability": prob,
                "bookmaker": "FIFA_RANKING_FALLBACK",
                "bookmaker_key": "fallback",
                "last_update": "",
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(RAW_DIR / "odds_outright.csv", index=False)
        logger.info(f"Saved {len(df)} fallback outright odds to {RAW_DIR / 'odds_outright.csv'}")
        _aggregate_outright_odds(df)

    return df


def _aggregate_outright_odds(df: pd.DataFrame):
    """Aggregate implied probabilities across bookmakers and save a summary."""
    if df.empty or "team" not in df.columns:
        return

    summary = (
        df.groupby("team")["implied_probability"]
        .mean()
        .reset_index()
        .rename(columns={"implied_probability": "avg_implied_probability"})
        .sort_values("avg_implied_probability", ascending=False)
        .reset_index(drop=True)
    )
    out_path = RAW_DIR / "odds_outright_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info(f"Saved outright odds summary to {out_path}")


def scrape_match_odds() -> pd.DataFrame:
    logger.info("Scraping per-match odds from ESPN...")

    fixtures_path = RAW_DIR / "wc2026_fixtures.csv"
    if not fixtures_path.exists():
        logger.warning("Fixtures file not found. Run scrape_world_cup_2026.py first.")
        return pd.DataFrame()

    fixtures = pd.read_csv(fixtures_path)
    all_odds = []

    for _, fixture in fixtures.iterrows():
        try:
            odds = _scrape_espn_match_odds(fixture)
            if odds:
                all_odds.append(odds)
        except Exception as e:
            logger.debug(f"Could not fetch odds for {fixture.get('home_team', '?')} vs {fixture.get('away_team', '?')}: {e}")

    if not all_odds:
        logger.warning("No match odds found from ESPN")
        return pd.DataFrame()

    df = pd.DataFrame(all_odds)
    out_path = RAW_DIR / "odds_match.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} match odds to {out_path}")

    return df


def _scrape_espn_match_odds(fixture: pd.Series) -> dict | None:
    home = fixture.get("home_team", "")
    away = fixture.get("away_team", "")

    search_url = "https://www.espn.com/soccer/schedule/_/league/FIFA.WORLD"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    try:
        resp = requests.get(search_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        odds_tables = soup.find_all("table", {"class": ["schedule", "Table"]})
        for table in odds_tables:
            rows = table.find_all("tr")
            for row in rows:
                text = row.get_text()
                if home in text or away in text:
                    odds_cells = row.find_all("td", class_=re.compile(r"odds|moneyline", re.I))
                    if odds_cells:
                        odds_texts = [cell.get_text(strip=True) for cell in odds_cells]
                        home_odds = _parse_american_odds(odds_texts[0]) if len(odds_texts) > 0 else None
                        draw_odds = _parse_american_odds(odds_texts[1]) if len(odds_texts) > 1 else None
                        away_odds = _parse_american_odds(odds_texts[2]) if len(odds_texts) > 2 else None

                        return {
                            "home_team": home,
                            "away_team": away,
                            "home_american_odds": home_odds,
                            "draw_american_odds": draw_odds,
                            "away_american_odds": away_odds,
                            "home_implied_prob": odds_to_probability(home_odds) if home_odds else None,
                            "draw_implied_prob": odds_to_probability(draw_odds) if draw_odds else None,
                            "away_implied_prob": odds_to_probability(away_odds) if away_odds else None,
                        }
    except requests.exceptions.RequestException:
        return None

    return None


def _parse_american_odds(text: str) -> int | None:
    text = text.strip().replace("+", "")
    try:
        return int(text)
    except ValueError:
        return None


if __name__ == "__main__":
    outright = fetch_outright_odds()
    if not outright.empty:
        print(f"Outright odds: {len(outright)} entries")
        print(outright.head(10))

    match_odds = scrape_match_odds()
    if not match_odds.empty:
        print(f"Match odds: {len(match_odds)} entries")
        print(match_odds.head(10))