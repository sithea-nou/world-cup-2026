import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    FIFA_RANKINGS_WIKI_PAGE,
    RAW_DIR,
    WIKIPEDIA_API_BASE,
)
from src.helpers import logger, normalize_team_name


def scrape_current_fifa_rankings() -> pd.DataFrame:
    logger.info("Scraping current FIFA rankings from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    rankings = []

    try:
        tables = pd.read_html(url, storage_options=headers)

        for table in tables:
            if table.shape[0] < 10:
                continue

            rank_col = None
            country_col = None
            points_col = None

            for col_idx in range(table.shape[1]):
                col_values = table.iloc[:, col_idx].dropna().astype(str)
                numeric_count = sum(1 for v in col_values if re.match(r"^\d+\.?\d*$", v.strip().replace(",", "")))
                if numeric_count >= 10 and rank_col is None:
                    rank_col = col_idx
                elif any(any(name in str(v).lower() for name in ["argentina", "brazil", "france", "germany"]) for v in col_values):
                    country_col = col_idx

            if rank_col is None or country_col is None:
                continue

            remaining_cols = [i for i in range(table.shape[1]) if i not in (rank_col, country_col)]
            for col_idx in remaining_cols:
                col_values = table.iloc[:, col_idx].dropna().astype(str)
                points_count = sum(1 for v in col_values if re.match(r"^\d{3,4}\.?\d*$", v.strip().replace(",", "")))
                if points_count >= 10:
                    points_col = col_idx
                    break

            for _, row in table.iterrows():
                try:
                    rank_val = str(row.iloc[rank_col]).strip()
                    rank_match = re.search(r"\d+", rank_val)
                    if not rank_match:
                        continue
                    rank = int(rank_match.group())

                    country = str(row.iloc[country_col]).strip()
                    country = re.sub(r"\[.*?\]", "", country).strip()
                    country = normalize_team_name(country)

                    if not country or len(country) < 2 or not country[0].isalpha():
                        continue
                    if any(w in country.lower() for w in ["ranking", "update", "change", "complete", "top 20"]):
                        continue

                    points = None
                    if points_col is not None:
                        points_val = str(row.iloc[points_col]).strip().replace(",", "").replace("\u202f", "")
                        points_match = re.match(r"^(\d+\.?\d*)", points_val)
                        if points_match:
                            try:
                                points = float(points_match.group(1))
                            except ValueError:
                                pass

                    rankings.append({"rank": rank, "country": country, "total_points": points})
                except (ValueError, IndexError, TypeError):
                    continue

            if len(rankings) >= 20:
                break

    except Exception as e:
        logger.warning(f"Failed to read Wikipedia tables: {e}")

    if not rankings:
        logger.warning("Could not parse FIFA rankings from Wikipedia tables, trying HTML fallback...")
        rankings = _scrape_fifa_rankings_html()

    if not rankings:
        logger.warning("All methods failed. Using historical rankings only.")
        return pd.DataFrame(columns=["rank", "country", "total_points"])

    df = pd.DataFrame(rankings)
    df = df.drop_duplicates(subset=["country"], keep="first")
    df = df.sort_values("rank").reset_index(drop=True)

    out_path = RAW_DIR / "fifa_rankings_current.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} FIFA rankings to {out_path}")

    return df


def _scrape_fifa_rankings_html() -> list[dict]:
    url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"
    headers = {"User-Agent": "WorldCupPredictor/1.0 (research project)"}

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    rankings = []
    for table in soup.find_all("table", {"class": ["wikitable", "wikitable sortable"]}):
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) < 3:
                continue

            try:
                rank_text = cols[0].get_text(strip=True)
                rank = int(re.sub(r"[^\d]", "", rank_text)) if rank_text else None
                if rank is None:
                    continue

                country_cell = cols[1]
                country = country_cell.get_text(strip=True)
                country = re.sub(r"\[.*?\]", "", country).strip()
                country = normalize_team_name(country)

                if not country or len(country) < 2:
                    continue

                points = None
                for col in cols[2:]:
                    text = col.get_text(strip=True).replace(",", "").replace("\u202f", "")
                    if re.match(r"^\d+\.?\d*$", text):
                        points = float(text)
                        break

                rankings.append(
                    {
                        "rank": rank,
                        "country": country,
                        "total_points": points,
                    }
                )
            except (ValueError, IndexError):
                continue

    return rankings


def merge_rankings(historical_path: Path, current_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging historical and current FIFA rankings...")

    if historical_path.exists():
        hist_df = pd.read_csv(historical_path)
        hist_df.columns = [c.strip().lower().replace(" ", "_") for c in hist_df.columns]

        if "rank_date" in hist_df.columns:
            hist_df["rank_date"] = pd.to_datetime(hist_df["rank_date"])
            latest_hist = hist_df["rank_date"].max()

            current_df["rank_date"] = pd.Timestamp.now()
            current_df["confederation"] = "Unknown"

            if "confederation" in hist_df.columns:
                confed_map = hist_df.drop_duplicates(subset=["country_full"])[
                    ["country_full", "confederation"
                ]
                ].values.tolist()
                confed_dict = {normalize_team_name(str(k)): v for k, v in confed_map}
                current_df["confederation"] = current_df["country"].map(confed_dict).fillna("Unknown")

            current_df = current_df.rename(
                columns={"country": "country_full", "total_points": "total_points", "rank": "rank"}
            )

            merged = pd.concat([hist_df, current_df], ignore_index=True)
            merged = merged.sort_values("rank_date").reset_index(drop=True)
        else:
            merged = hist_df
    else:
        merged = current_df

    out_path = RAW_DIR / "fifa_rankings_merged.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Merged rankings saved to {out_path} ({len(merged)} rows)")

    return merged


if __name__ == "__main__":
    current = scrape_current_fifa_rankings()
    hist_path = RAW_DIR / "fifa_rankings" / "fifa_ranking.csv"
    if hist_path.exists():
        merge_rankings(hist_path, current)
    else:
        logger.warning(f"Historical rankings not found at {hist_path}. Run download_kaggle.py first.")