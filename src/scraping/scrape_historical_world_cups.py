import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR, WIKIPEDIA_API_BASE
from src.helpers import logger, normalize_team_name


WORLD_CUP_YEARS = list(range(1930, 2023, 4))
WORLD_CUP_YEARS = [y for y in WORLD_CUP_YEARS if y not in [1942, 1946]]


def scrape_historical_brackets() -> pd.DataFrame:
    logger.info("Scraping historical World Cup brackets from Wikipedia...")

    all_records = []

    for year in WORLD_CUP_YEARS:
        try:
            records = _scrape_single_world_cup(year)
            if records:
                all_records.extend(records)
                logger.info(f"  {year}: {len(records)} matches")
        except Exception as e:
            logger.warning(f"  {year}: Failed - {e}")
            continue

    df = pd.DataFrame(all_records)

    if df.empty:
        logger.warning("No historical brackets scraped, falling back to Kaggle data filter")
        return _get_historical_from_kaggle()

    out_path = RAW_DIR / "historical_world_cups.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} historical WC matches to {out_path}")

    return df


def _scrape_single_world_cup(year: int) -> list[dict]:
    """Scrape knockout-stage match results from a Wikipedia World Cup page."""
    page_name = f"{year}_FIFA_World_Cup"
    url = f"https://en.wikipedia.org/wiki/{page_name}"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    # Try modern bracket tables (class "wikitable")
    bracket_tables = soup.find_all("table", {"class": "wikitable"})

    records = []

    for table in bracket_tables:
        round_type = _detect_round_type(table)
        if round_type is None:
            continue

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:
                continue

            text = " ".join(cell.get_text(" ", strip=True) for cell in cells)
            if not re.search(r"\d+\s*[-–]\s*\d+", text):
                continue

            teams = _extract_teams_from_cells(cells)
            if len(teams) < 2:
                continue

            score_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
            if not score_match:
                continue

            home_score = int(score_match.group(1))
            away_score = int(score_match.group(2))

            # Avoid duplicates: skip rows where both teams are identical
            if teams[0].lower() == teams[1].lower():
                continue

            records.append(
                {
                    "year": year,
                    "round": round_type,
                    "home_team": teams[0],
                    "away_team": teams[1],
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

    if not records:
        # Fallback 1: parse raw wikitext via index.php (avoids API rate limits)
        records = _scrape_wikitext_raw(year)

    if not records:
        # Fallback 2: parse wikitext via API (with a polite delay)
        time.sleep(0.5)
        records = _scrape_wikitext_world_cup(year)

    return records


def _detect_round_type(table) -> str | None:
    """Detect the tournament round from a table's caption, header, or preceding heading."""
    # Check caption
    caption = table.find("caption")
    if caption:
        text = caption.get_text(" ", strip=True).lower()
        for rnd in ["round of 16", "quarter-finals", "semi-finals", "final", "third place"]:
            if rnd.replace("-", " ") in text or rnd in text:
                return rnd.replace(" ", "-") if " " in rnd else rnd

    # Check preceding heading
    prev = table.find_previous(["h2", "h3", "h4"])
    if prev:
        text = prev.get_text(" ", strip=True).lower()
        for rnd in ["round of 16", "quarter-finals", "semi-finals", "final", "third place"]:
            if rnd.replace("-", " ") in text or rnd in text:
                return rnd.replace(" ", "-") if " " in rnd else rnd

    # Check headers
    headers = table.find_all(["th"])
    for header in headers:
        text = header.get_text(" ", strip=True).lower()
        for rnd in ["round of 16", "quarter-finals", "semi-finals", "final", "third place"]:
            if rnd.replace("-", " ") in text or rnd in text:
                return rnd.replace(" ", "-") if " " in rnd else rnd

    return None


def _extract_teams_from_cells(cells) -> list[str]:
    """Extract team names from table cells."""
    teams = []
    for cell in cells:
        # Try flag icon alt text / links first
        links = cell.find_all("a")
        for link in links:
            title = link.get("title", "")
            text = link.get_text(strip=True)
            name = text if text else title
            name = normalize_team_name(re.sub(r"\[.*?\]", "", name).strip())
            if name and len(name) > 1 and name[0].isalpha() and name not in teams:
                teams.append(name)
                if len(teams) >= 2:
                    return teams[:2]

    # Fallback: plain text parsing
    if len(teams) < 2:
        all_text = " ".join(cell.get_text(" ", strip=True) for cell in cells)
        # Remove score patterns and parentheticals
        cleaned = re.sub(r"\d+\s*[-–]\s*\d+", "", all_text)
        cleaned = re.sub(r"\(.*?\)", "", cleaned)
        parts = [p.strip() for p in cleaned.split() if len(p.strip()) > 1 and p.strip()[0].isalpha()]
        # Heuristic: first two plausible team tokens
        for part in parts[:2]:
            name = normalize_team_name(part)
            if name and name not in teams:
                teams.append(name)

    return teams[:2]


def _scrape_wikitext_raw(year: int) -> list[dict]:
    """Scrape raw wikitext via index.php (avoids API rate limits)."""
    page_name = f"{year}_FIFA_World_Cup"
    url = f"https://en.wikipedia.org/w/index.php?title={page_name}&action=raw"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        return []

    wikitext = resp.text
    if not wikitext:
        return []

    return _parse_wikitext_lines(wikitext, year)


def _parse_wikitext_lines(wikitext: str, year: int) -> list[dict]:
    records = []
    round_type = "group"

    for line in wikitext.split("\n"):
        if re.search(r"===?\s*(Round of \d+|Quarter-finals?|Semi-finals?|Final|Third.*place)\s*===?", line, re.IGNORECASE):
            round_name = re.search(
                r"===?\s*(Round of \d+|Quarter-finals?|Semi-finals?|Final|Third.*place)\s*===?", line, re.IGNORECASE
            )
            round_type = round_name.group(1).lower().replace("-", " ") if round_name else "knockout"
            continue

        if re.search(r"===?\s*Group\s+([A-H])\s*===?", line, re.IGNORECASE):
            round_type = "group"
            continue

        if round_type == "group":
            continue

        score_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", line)
        if score_match and "|" in line:
            teams = re.findall(r"\[\[([^\]]+)\]\]", line)
            if len(teams) >= 2:
                home = normalize_team_name(teams[0].split("|")[-1])
                away = normalize_team_name(teams[1].split("|")[-1])

                records.append(
                    {
                        "year": year,
                        "round": round_type,
                        "home_team": home,
                        "away_team": away,
                        "home_score": int(score_match.group(1)),
                        "away_score": int(score_match.group(2)),
                    }
                )

    return records


def _scrape_wikitext_world_cup(year: int) -> list[dict]:
    page_name = f"{year}_FIFA_World_Cup"
    url = f"{WIKIPEDIA_API_BASE}?action=parse&page={page_name}&prop=wikitext&format=json"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    if not wikitext:
        return []

    return _parse_wikitext_lines(wikitext, year)


def _get_historical_from_kaggle() -> pd.DataFrame:
    results_path = RAW_DIR / "international_matches" / "results.csv"
    if not results_path.exists():
        logger.error("Kaggle results not found. Run download_kaggle.py first.")
        return pd.DataFrame()

    df = pd.read_csv(results_path)
    wc_df = df[df["tournament"] == "FIFA World Cup"].copy()
    wc_df = wc_df.rename(
        columns={
            "date": "date",
            "home_team": "home_team",
            "away_team": "away_team",
            "home_score": "home_score",
            "away_score": "away_score",
        }
    )
    wc_df["year"] = pd.to_datetime(wc_df["date"]).dt.year
    wc_df["round"] = "unknown"
    wc_df = wc_df[["year", "round", "home_team", "away_team", "home_score", "away_score"]]

    out_path = RAW_DIR / "historical_world_cups.csv"
    wc_df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(wc_df)} historical WC matches from Kaggle to {out_path}")

    return wc_df


if __name__ == "__main__":
    df = scrape_historical_brackets()
    print(f"Historical WC matches: {len(df)}")
    if not df.empty:
        print(df.groupby("year").size())