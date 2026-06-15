import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    RAW_DIR,
    WC2026_WIKI_PAGE,
    WIKIPEDIA_API_BASE,
)
from src.helpers import logger, normalize_team_name


def scrape_wc2026_groups() -> pd.DataFrame:
    logger.info("Scraping 2026 FIFA World Cup groups from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    try:
        tables = pd.read_html(url, storage_options=headers)
    except Exception as e:
        logger.warning(f"Failed to read Wikipedia tables: {e}")
        return pd.DataFrame(columns=["group", "team", "pot"])

    groups = []
    current_group = None
    group_idx = 0
    group_letters = list("ABCDEFGHIJKL")

    for table in tables:
        cols_lower = [str(c).lower() for c in table.columns]

        has_pos = any("pos" in c for c in cols_lower)
        has_team = any("team" in c for c in cols_lower)

        if not (has_pos and has_team):
            continue

        if table.shape[0] < 3 or table.shape[0] > 6:
            continue

        if group_idx >= 12:
            break

        current_group = group_letters[group_idx]

        team_col = None
        for i, c in enumerate(table.columns):
            if "team" in str(c).lower() or "vte" in str(c).lower():
                team_col = i
                break

        if team_col is None:
            team_col = 1

        for _, row in table.iterrows():
            team_name = str(row.iloc[team_col]).strip()
            team_name = re.sub(r"\[.*?\]", "", team_name).strip()
            team_name = re.sub(r"\(H\)", "", team_name).strip()
            team_name = re.sub(r"\s+", " ", team_name).strip()

            if not team_name or len(team_name) < 2 or not team_name[0].isalpha():
                continue
            if any(w in team_name.lower() for w in ["advance", "possible", "ranking", "group"]):
                continue

            team_name = normalize_team_name(team_name)

            pot = len([r for r in groups if r["group"] == current_group]) + 1
            groups.append({"group": current_group, "team": team_name, "pot": pot})

        group_idx += 1

    if not groups:
        logger.warning("Could not parse groups from Wikipedia tables.")
        return pd.DataFrame(columns=["group", "team", "pot"])

    df = pd.DataFrame(groups)
    df = df.drop_duplicates(subset=["group", "team"], keep="first").reset_index(drop=True)

    for group_letter in df["group"].unique():
        group_teams = df[df["group"] == group_letter].reset_index(drop=True)
        for i, idx in enumerate(group_teams.index):
            df.at[idx, "pot"] = i + 1

    out_path = RAW_DIR / "wc2026_groups.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} group entries to {out_path}")

    return df


def _scrape_groups_html() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    records = []
    group_letter = None

    for element in soup.find_all(["h3", "table"]):
        if element.name == "h3":
            span = element.find("span", {"id": re.compile(r"^Group_[A-L]$")})
            if span:
                group_letter = span["id"].split("_")[-1]
                continue

        if element.name == "table" and group_letter:
            rows = element.find_all("tr")
            pot = 1
            for row in rows[1:]:
                cols = row.find_all("td")
                if not cols:
                    continue
                team_cell = cols[0] if len(cols) > 0 else None
                if team_cell:
                    team_name = team_cell.get_text(strip=True)
                    team_name = normalize_team_name(re.sub(r"\[.*?\]", "", team_name))
                    if team_name and len(team_name) > 2:
                        records.append(
                            {"group": group_letter, "team": team_name, "pot": pot}
                        )
                        pot += 1
            group_letter = None

    return pd.DataFrame(records)


def scrape_wc2026_fixtures() -> pd.DataFrame:
    logger.info("Scraping 2026 FIFA World Cup fixtures from Wikipedia...")

    groups_df = pd.read_csv(RAW_DIR / "wc2026_groups.csv") if (RAW_DIR / "wc2026_groups.csv").exists() else pd.DataFrame()

    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    try:
        tables = pd.read_html(url, storage_options=headers)
    except Exception as e:
        logger.warning(f"Failed to read Wikipedia tables: {e}")
        return pd.DataFrame()

    fixtures = []
    match_num = 0

    group_letter_idx = 0
    group_letters = list("ABCDEFGHIJKL")

    for table in tables:
        cols_lower = [str(c).lower() for c in table.columns]
        has_pos = any("pos" in c for c in cols_lower)
        has_team = any("team" in c or "vte" in c for c in cols_lower)
        has_pld = any("pld" in c for c in cols_lower)

        if not (has_pos and has_team and has_pld):
            continue

        if table.shape[0] < 3 or table.shape[0] > 5:
            continue

        if group_letter_idx >= 12:
            break

        group_letter = group_letters[group_letter_idx]
        team_col = None
        for i, c in enumerate(table.columns):
            if "team" in str(c).lower() or "vte" in str(c).lower():
                team_col = i
                break
        if team_col is None:
            team_col = 1

        teams = []
        for _, row in table.iterrows():
            team_name = str(row.iloc[team_col]).strip()
            team_name = re.sub(r"\[.*?\]", "", team_name).strip()
            team_name = re.sub(r"\(H\)", "", team_name).strip()
            team_name = normalize_team_name(team_name)
            if team_name and len(team_name) > 2 and team_name[0].isalpha():
                if not any(w in team_name.lower() for w in ["advance", "possible", "ranking", "group"]):
                    teams.append(team_name)

        if len(teams) >= 4:
            from itertools import combinations
            for home, away in combinations(teams[:4], 2):
                match_num += 1
                fixtures.append({
                    "match_number": match_num,
                    "date": None,
                    "home_team": home,
                    "away_team": away,
                    "group": group_letter,
                    "venue": None,
                    "city": None,
                })

        group_letter_idx += 1

    if not fixtures and not groups_df.empty:
        logger.warning("Could not parse fixtures, generating from groups...")
        from itertools import combinations
        for group_letter in groups_df["group"].unique():
            teams = groups_df[groups_df["group"] == group_letter]["team"].tolist()
            for home, away in combinations(teams, 2):
                match_num += 1
                fixtures.append({
                    "match_number": match_num,
                    "date": None,
                    "home_team": home,
                    "away_team": away,
                    "group": group_letter,
                    "venue": None,
                    "city": None,
                })

    df = pd.DataFrame(fixtures)
    out_path = RAW_DIR / "wc2026_fixtures.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} fixtures to {out_path}")

    return df


if __name__ == "__main__":
    groups = scrape_wc2026_groups()
    print(f"Groups: {len(groups)} entries")
    print(groups.head(20))

    fixtures = scrape_wc2026_fixtures()
    print(f"Fixtures: {len(fixtures)} matches")
    print(fixtures.head(10))