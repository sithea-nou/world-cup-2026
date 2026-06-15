import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR
from src.helpers import logger, normalize_team_name


ESPN_WC_URL = "https://www.espn.com/soccer/schedule/_/league/FIFA.WORLD/cup/2026"
FIFA_WC_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026"
WIKIPEDIA_WC_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"


def scrape_live_results() -> pd.DataFrame:
    logger.info("Scraping live WC 2026 match results...")

    results = _scrape_espn_results()
    if results.empty:
        results = _scrape_wikipedia_results()

    manual_path = RAW_DIR / "wc2026_results_manual.csv"
    if manual_path.exists():
        manual_df = pd.read_csv(manual_path)
        if not manual_df.empty and not manual_df.dropna(how="all").empty:
            has_scores = manual_df["home_score"].notna() & manual_df["away_score"].notna()
            manual_played = manual_df[has_scores].copy() if "home_score" in manual_df.columns else pd.DataFrame()

            if not manual_played.empty and not results.empty:
                results = _merge_with_manual(results, manual_played)
            elif not manual_played.empty:
                results = manual_played

    out_path = RAW_DIR / "wc2026_results_live.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info(f"Saved {len(results)} live results to {out_path}")

    return results


def _scrape_espn_results() -> pd.DataFrame:
    logger.info("  Trying ESPN...")
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    try:
        resp = requests.get(ESPN_WC_URL, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.debug(f"  ESPN request failed: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    records = []

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            date_text = cols[0].get_text(strip=True)
            home_team = normalize_team_name(cols[1].get_text(strip=True))
            away_team = normalize_team_name(cols[2].get_text(strip=True))
            score_text = cols[3].get_text(strip=True)

            score_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", score_text)
            if score_match:
                home_score = int(score_match.group(1))
                away_score = int(score_match.group(2))
            else:
                home_score = None
                away_score = None

            try:
                date = pd.to_datetime(date_text)
            except (ValueError, TypeError):
                date = None

            records.append(
                {
                    "date": date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "source": "espn",
                }
            )

    return pd.DataFrame(records)


def _scrape_wikipedia_results() -> pd.DataFrame:
    logger.info("  Trying Wikipedia (HTML)...")
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    try:
        resp = requests.get(WIKIPEDIA_WC_URL, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.debug(f"  Wikipedia request failed: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    records = []

    group_spans = soup.find_all("span", class_="mw-headline", string=re.compile(r"^Group [A-L]$"))
    for group_span in group_spans:
        group_letter = re.search(r"Group ([A-L])", group_span.get_text()).group(1)
        parent_heading = group_span.find_parent(["h2", "h3"])
        if not parent_heading:
            continue

        next_el = parent_heading.find_next_sibling()
        while next_el and next_el.name != "table":
            if next_el.name in ("h2", "h3"):
                break
            next_el = next_el.find_next_sibling()

        if not next_el or next_el.name != "table":
            continue

        wikitable = next_el
        rows = wikitable.find_all("tr")
        if len(rows) < 2:
            continue

        header_row = rows[0]
        col_texts = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

        team_col = None
        for idx, text in enumerate(col_texts):
            if text.lower() in ("team", "teams"):
                team_col = idx
                break

        if team_col is None:
            for idx, th in enumerate(header_row.find_all("th")):
                if th.get("scope", "") == "col":
                    team_col = idx
                    break

        if team_col is None:
            continue

        match_cols = []
        for idx, text in enumerate(col_texts):
            if idx != team_col and text.strip():
                match_cols.append(idx)

        team_names = []
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) <= team_col:
                continue
            team = cells[team_col].get_text(strip=True)
            team = re.sub(r"\s*\(.*\)\s*", "", team)
            team = team.strip()
            if not team or team.lower().startswith("rank") or team.lower().startswith("team"):
                continue
            team_names.append(team)

        if len(team_names) < 4:
            continue

        for row_idx, row in enumerate(rows[1:]):
            if row_idx >= len(team_names):
                break
            cells = row.find_all(["th", "td"])
            home_team = team_names[row_idx]

            for col_idx in match_cols:
                if col_idx >= len(cells):
                    continue
                cell_text = cells[col_idx].get_text(strip=True)
                score_match = re.match(r"(\d+)\s*[–\-]\s*(\d+)", cell_text)
                if not score_match:
                    continue

                opp_row_idx = col_idx
                if opp_row_idx >= len(team_names) or opp_row_idx == row_idx:
                    opp_row_idx = col_idx - (1 if col_idx > team_col else 0)

                if opp_row_idx < 0 or opp_row_idx >= len(team_names) or opp_row_idx == row_idx:
                    continue

                away_team = team_names[opp_row_idx]
                home_score = int(score_match.group(1))
                away_score = int(score_match.group(2))

                key = (group_letter, home_team, away_team)
                if key not in {(r["group"], r["home_team"], r["away_team"]) for r in records}:
                    records.append(
                        {
                            "date": None,
                            "home_team": normalize_team_name(home_team),
                            "away_team": normalize_team_name(away_team),
                            "home_score": home_score,
                            "away_score": away_score,
                            "group": group_letter,
                            "source": "wikipedia",
                        }
                    )

    df = pd.DataFrame(records)
    if not df.empty and "group" in df.columns:
        df = df.sort_values(["group", "home_team"]).reset_index(drop=True)

    return df


def _merge_with_manual(auto_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    manual_teams = set(zip(manual_df["home_team"], manual_df["away_team"]))
    auto_teams = set(zip(auto_df["home_team"], auto_df["away_team"]))

    overlap = manual_teams & auto_teams
    if overlap:
        mask = ~auto_df.apply(lambda r: (r["home_team"], r["away_team"]) in overlap, axis=1)
        auto_only = auto_df[mask]
    else:
        auto_only = auto_df

    merged = pd.concat([manual_df, auto_only], ignore_index=True)

    if "date" in merged.columns:
        merged = merged.sort_values("date", na_position="last").reset_index(drop=True)

    return merged


if __name__ == "__main__":
    results = scrape_live_results()
    print(f"Live results: {len(results)} matches")
    played = results[results["home_score"].notna()] if not results.empty else results
    print(f"Played (with scores): {len(played)} matches")
    if not played.empty:
        print(played.head(10))