import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR
from src.helpers import logger, normalize_team_name

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

TRANSFERMARKT_SEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={query}&x=0&y=0"
TRANSFERMARKT_TEAM_URL = "https://www.transfermarkt.com{slug}/startseite/verein/{club_id}/saison_id/2025"

TEAM_SLUG_OVERRIDES = {
    "United States": "us-usa",
    "South Korea": "sudkorea",
    "Ivory Coast": "elfenbeinkuste",
    "Bosnia-Herzegovina": "bosnien-herzegovina",
    "Czech Republic": "tschechien",
    "Scotland": "schottland",
    "Wales": "wales",
    "Morocco": "marokko",
    "Switzerland": "schweiz",
    "Netherlands": "niederlande",
    "Croatia": "kroatien",
    "Turkey": "turkei",
    "Poland": "polen",
    "Sweden": "schweden",
    "Denmark": "danemark",
    "Spain": "spanien",
    "Italy": "italien",
    "France": "frankreich",
    "Germany": "deutschland",
    "Portugal": "portugal",
    "Brazil": "brasilien",
    "Argentina": "argentinien",
    "Colombia": "kolumbien",
    "Ecuador": "ecuador",
    "Paraguay": "paraguay",
    "Uruguay": "uruguay",
    "Chile": "chile",
    "Peru": "peru",
    "Mexico": "mexiko",
    "Canada": "kanada",
    "Costa Rica": "costa-rica",
    "Jamaica": "jamaika",
    "Panama": "panama",
    "Honduras": "honduras",
    "Haiti": "haiti",
    "Curaçao": "curacao",
    "Japan": "japan",
    "Iran": "iran",
    "Saudi Arabia": "saudi-arabien",
    "Australia": "australien",
    "Qatar": "katar",
    "Uzbekistan": "usbekistan",
    "Iraq": "irak",
    "Jordan": "jordanien",
    "United Arab Emirates": "vereinigte-arabische-emirate",
    "Oman": "oman",
    "China": "china",
    "New Zealand": "neuseeland",
    "Algeria": "algerien",
    "Tunisia": "tunesien",
    "Egypt": "agypten",
    "Senegal": "senegal",
    "Nigeria": "nigeria",
    "Ghana": "ghana",
    "Cameroon": "kamerun",
    "Mali": "mali",
    "Guinea": "guinea",
    "Congo DR": "demokratische-republik-kongo",
    "Burkina Faso": "burkina-faso",
    "South Africa": "sudafrika",
    "Cape Verde": "kap-verde",
    "Zambia": "sambia",
    "Kenya": "kenia",
    "Norway": "norwegen",
    "Austria": "osterreich",
    "Hungary": "ungarn",
    "Romania": "rumaenien",
    "Slovakia": "slowakei",
    "Slovenia": "slowenien",
    "Greece": "griechenland",
    "Serbia": "serbien",
    "Ukraine": "ukraine",
    "Russia": "russland",
    "Republic of Ireland": "irland",
    "Finland": "finnland",
    "Iceland": "island",
    "Georgia": "georgien",
    "Albania": "albanien",
    "Montenegro": "montenegro",
    "FYR Macedonia": "mazedonien",
    "Israel": "israel",
    "Bahrain": "bahrain",
    "Syria": "syrien",
    "Kuwait": "kuwait",
    "Thailand": "thailand",
    "Vietnam": "vietnam",
    "India": "indien",
    "Lebanon": "libanon",
    "Palestine": "palaestina",
    "Tajikistan": "tadschikistan",
    "North Korea": "nordkorea",
    "Fiji": "fidschi",
    "Papua New Guinea": "papua-neuguinea",
    "Solomon Islands": "salomonen",
    "Tahiti": "tahiti",
    "Suriname": "suriname",
    "Guyana": "guyana",
    "Venezuela": "venezuela",
    "Bolivia": "bolivien",
}


TEAM_SEARCH_OVERRIDES = {
    "Congo DR": "Democratic Republic of the Congo",
    "Ivory Coast": "Ivory Coast",
    "United States": "USA",
}


def _find_national_team_id(team_name: str) -> tuple[str, int] | None:
    search_name = TEAM_SEARCH_OVERRIDES.get(team_name, team_name)
    slug = TEAM_SLUG_OVERRIDES.get(team_name, team_name.lower().replace(" ", "-"))
    search_url = TRANSFERMARKT_SEARCH_URL.format(query=search_name.replace(" ", "+"))

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=re.compile(r"/startseite/verein/\d+")):
            href = a.get("href", "")
            text = a.text.strip()
            if not text or "verein" not in href:
                continue

            is_national = any(
                x in href.lower()
                for x in [f"/{slug}/", f"/{slug.lower()}/"]
            )
            if not is_national:
                is_national = text == team_name

            has_youth = any(
                x in href.lower()
                for x in ["-u", "u20", "u21", "u23", "u19", "u18", "u17", "u16", "u15", "jugend"]
            )
            if is_national and not has_youth:
                match = re.search(r"/verein/(\d+)", href)
                if match:
                    club_id = int(match.group(1))
                    path_match = re.match(r"(.+)/startseite", href)
                    path_slug = path_match.group(1) if path_match else f"/{slug}"
                    return path_slug, club_id

    except Exception as e:
        logger.warning(f"Search failed for {team_name}: {e}")

    return None


def _parse_market_value(value_str: str) -> float:
    value_str = value_str.strip().replace(",", "")
    if "€" not in value_str:
        return 0.0
    value_str = value_str.replace("€", "").strip()
    if "bn" in value_str.lower():
        return float(re.sub(r"[^\d.]", "", value_str.lower().split("bn")[0])) * 1000.0
    elif "m" in value_str.lower():
        num_part = value_str.lower().split("m")[0].strip()
        return float(re.sub(r"[^\d.]", "", num_part))
    elif "k" in value_str.lower():
        num_part = value_str.lower().split("k")[0].strip()
        return float(re.sub(r"[^\d.]", "", num_part)) / 1000.0
    return 0.0


def _scrape_team_squad_data(slug: str, club_id: int) -> dict | None:
    url = TRANSFERMARKT_TEAM_URL.format(slug=slug, club_id=club_id)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        market_value_m = 0.0
        squad_size = 0
        avg_age = 0.0
        foreigners = 0

        mv_tag = soup.find("a", class_=lambda c: c and "market-value" in c)
        if mv_tag:
            market_value_m = _parse_market_value(mv_tag.text)

        for label in soup.find_all("li", class_="data-header__label"):
            text = label.get_text(strip=True)
            content = label.find("span", class_="data-header__content")
            if not content:
                continue
            val = content.get_text(strip=True)
            if "Squad size" in text:
                nums = re.findall(r"\d+", val)
                if nums:
                    squad_size = int(nums[0])
            elif "Average age" in text:
                ages = re.findall(r"[\d.]+", val)
                if ages:
                    avg_age = float(ages[0])
            elif "Foreigners" in text:
                match = re.match(r"(\d+)\s", val) or re.match(r"^(\d{1,2})", val)
                if match:
                    foreigners = int(match.group(1))

        player_values = []
        table = soup.select_one(".responsive-table > .grid-view > .items > tbody")
        if table:
            for row in table.find_all("tr", class_=re.compile(r"^(even|odd)$")):
                cells = row.find_all("td")
                for cell in cells:
                    text = cell.get_text(strip=True)
                    if "€" in text:
                        pv = _parse_market_value(text)
                        if pv > 0:
                            player_values.append(pv)
                        break

        avg_player_value = sum(player_values) / len(player_values) if player_values else 0.0
        top_player_value = max(player_values) if player_values else 0.0

        return {
            "squad_market_value_m": market_value_m,
            "squad_size": squad_size,
            "avg_age": avg_age,
            "foreigners": foreigners,
            "avg_player_value_m": avg_player_value,
            "top_player_value_m": top_player_value,
        }

    except Exception as e:
        logger.warning(f"Scrape failed for {slug}/{club_id}: {e}")
        return None


def scrape_squad_quality(teams: list[str] | None = None) -> pd.DataFrame:
    if teams is None:
        groups_path = RAW_DIR / "wc2026_groups.csv"
        if groups_path.exists():
            teams_df = pd.read_csv(groups_path)
            teams = teams_df["team"].unique().tolist()
        else:
            logger.error("No teams list provided and wc2026_groups.csv not found")
            return pd.DataFrame()

    logger.info(f"Scraping squad quality for {len(teams)} teams from Transfermarkt...")

    results = []
    for i, team in enumerate(teams):
        normalized = normalize_team_name(team)
        logger.info(f"  [{i + 1}/{len(teams)}] {team} (normalized: {normalized})")

        team_id = _find_national_team_id(normalized)
        if team_id is None:
            team_id = _find_national_team_id(team)
        if team_id is None:
            logger.warning(f"  Could not find Transfermarkt ID for {team}")
            results.append({"team": team, "squad_market_value_m": 0.0, "squad_size": 0,
                           "avg_age": 0.0, "foreigners": 0, "avg_player_value_m": 0.0,
                           "top_player_value_m": 0.0})
            time.sleep(1)
            continue

        slug, club_id = team_id
        logger.info(f"  Found: slug={slug}, club_id={club_id}")

        data = _scrape_team_squad_data(slug, club_id)
        time.sleep(2)

        if data is None:
            logger.warning(f"  Failed to scrape squad data for {team}")
            results.append({"team": team, "squad_market_value_m": 0.0, "squad_size": 0,
                           "avg_age": 0.0, "foreigners": 0, "avg_player_value_m": 0.0,
                           "top_player_value_m": 0.0})
            continue

        results.append({"team": team, **data})
        logger.info(f"  MV=€{data['squad_market_value_m']:.1f}m, Squad={data['squad_size']}, "
                    f"AvgVal=€{data['avg_player_value_m']:.2f}m")

    df = pd.DataFrame(results)
    out_path = RAW_DIR / "squad_quality.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Squad quality data saved to {out_path} ({len(df)} teams)")

    return df


if __name__ == "__main__":
    df = scrape_squad_quality()
    if not df.empty:
        print("\nSquad Quality Summary:")
        print(df.sort_values("squad_market_value_m", ascending=False).to_string(index=False))