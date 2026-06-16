# Data Sources

This document describes all data sources used by the World Cup 2026 ML Predictor, including access methods, formats, columns, and update frequencies.

## Overview

| Source | Type | Primary Use | Access |
|--------|------|-------------|--------|
| Kaggle (martj42) | International match results | Historical features, Elo computation | Kaggle API |
| Kaggle (cashncarry) | FIFA rankings | FIFA rank/points features | Kaggle API |
| Wikipedia (2026 FIFA World Cup) | Groups, fixtures | WC2026 simulation input | HTTP scrape |
| Wikipedia (FIFA Men's World Ranking) | Current FIFA rankings | Ranking features | HTTP scrape |
| ESPN | Per-match odds | Odds features | HTTP scrape |
| the-odds-api | Outright tournament odds | Odds features, visualization | REST API |
| Wikipedia (historical WC pages) | Historical brackets | Validation, analysis | HTTP scrape |
| Manual CSV | Live results override | Live validation, retraining | File |
| Continents CSV | Confederation mapping | Confederation features | Static file |

---

## Kaggle: International Football Results

- **Dataset**: `martj42/international-football-results-from-1872-to-2017`
- **URL**: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- **Format**: CSV (downloaded via Kaggle API, extracted from ZIP)
- **Access**: `src/scraping/download_kaggle.py` -- `download_match_results()`
- **Output**: `data/raw/international_matches/results.csv`
- **Rows**: 48,000+ (1872 to present)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | string (YYYY-MM-DD) | Match date |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `home_score` | int | Home team goals |
| `away_score` | int | Away team goals |
| `tournament` | string | Tournament name (e.g., "FIFA World Cup", "Friendly") |
| `city` | string | Match city |
| `country` | string | Match country |
| `neutral` | string | "True" if neutral venue |

### Additional Files

| File | Description |
|------|-------------|
| `shootouts.csv` | Penalty shootout results |
| `former_names.csv` | Historical country name mappings |

### Update Frequency

The Kaggle dataset is updated periodically by the maintainer. Re-download by running:

```bash
python run_pipeline.py --step scraping
```

Or force re-download by deleting `data/raw/international_matches/` first.

---

## Kaggle: FIFA World Ranking

- **Dataset**: `cashncarry/fifaworldranking`
- **URL**: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
- **Format**: CSV (downloaded via Kaggle API)
- **Access**: `src/scraping/download_kaggle.py` -- `download_fifa_rankings()`
- **Output**: `data/raw/fifa_rankings/fifa_ranking.csv`
- **Rows**: 60,000+ (1993 to present)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `rank` | int | FIFA ranking position |
| `country_full` | string | Full country name |
| `country_abrv` | string | 3-letter country code |
| `total_points` | float | Total ranking points |
| `rank_date` | string (YYYY-MM-DD) | Date of ranking |

### Update Frequency

Downloaded alongside match results. The historical rankings are supplemented with current rankings from Wikipedia (see below).

---

## Wikipedia: WC2026 Groups and Fixtures

- **Source**: Wikipedia page "2026 FIFA World Cup"
- **URL**: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup
- **Format**: HTML / Wikitext (parsed via MediaWiki API)
- **Access**: `src/scraping/scrape_world_cup_2026.py`
- **Output**: `data/raw/wc2026_groups.csv`, `data/raw/wc2026_fixtures.csv`

### Groups CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `group` | string | Group letter (A-L) |
| `team` | string | Team name (normalized) |
| `pot` | int | Pot number (1-4) for draw |

### Fixtures CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `match_number` | int | Sequential match number |
| `date` | string/datetime | Scheduled match date |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `group` | string | Group letter (if group stage) |
| `venue` | string | Stadium name (may be null) |
| `city` | string | Host city (may be null) |

### Access Method

The scraper first attempts to parse groups and fixtures from the Wikipedia API wikitext. If that fails, it falls back to HTML parsing with BeautifulSoup. Team names are normalized using `TEAM_NAME_MAPPING` in `config.py`.

### Update Frequency

Re-scrape whenever group compositions or fixtures change (e.g., after final draw). WC2026 groups may be updated as qualifying completes.

---

## Wikipedia: Current FIFA Rankings

- **Source**: Wikipedia page "FIFA Men's World Ranking"
- **URL**: https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking
- **Format**: HTML (parsed with BeautifulSoup)
- **Access**: `src/scraping/scrape_fifa_rankings.py` -- `scrape_current_fifa_rankings()`
- **Output**: `data/raw/fifa_rankings_current.csv`

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `rank` | int | Current FIFA ranking position |
| `country` | string | Country name (normalized) |
| `total_points` | float | Current ranking points |

### Merging

`merge_rankings()` combines historical Kaggle rankings with current Wikipedia rankings:

- Historical data (`fifa_ranking.csv`) + current data (`fifa_rankings_current.csv`) -> `fifa_rankings_merged.csv`
- Current rankings are timestamped with the scrape date
- Confederation information is propagated from historical data

---

## ESPN: Per-Match Odds

- **Source**: ESPN soccer schedule pages
- **URL**: https://www.espn.com/soccer/schedule/_/league/FIFA.WORLD
- **Format**: HTML (parsed with BeautifulSoup)
- **Access**: `src/scraping/scrape_odds.py` -- `scrape_match_odds()`
- **Output**: `data/raw/odds_match.csv`

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `home_american_odds` | int/null | American odds for home win |
| `draw_american_odds` | int/null | American odds for draw |
| `away_american_odds` | int/null | American odds for away win |
| `home_implied_prob` | float/null | Implied home win probability |
| `draw_implied_prob` | float/null | Implied draw probability |
| `away_implied_prob` | float/null | Implied away win probability |

### American Odds to Probability

```python
def odds_to_probability(american_odds):
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)
```

---

## the-odds-api: Outright Odds

- **Source**: the-odds-api.com
- **URL**: https://api.the-odds-api.com/v4/sports/football/world_cup/odds/
- **Format**: JSON API
- **Access**: `src/scraping/scrape_odds.py` -- `fetch_outright_odds()`
- **Output**: `data/raw/odds_outright.csv`, `data/raw/odds_outright.json`
- **Requires**: `ODDS_API_KEY` environment variable (free tier available at https://the-odds-api.com/)

### CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `team` | string | Team name |
| `american_odds` | int | American odds for tournament winner |
| `implied_probability` | float | Implied probability of winning |
| `bookmaker` | string | Bookmaker name |
| `bookmaker_key` | string | Bookmaker identifier |
| `last_update` | string | Last update timestamp |

### API Parameters

| Parameter | Value |
|-----------|-------|
| `apiKey` | From `ODDS_API_KEY` env var |
| `regions` | `us` |
| `markets` | `outright` |
| `oddsFormat` | `american` |

### Rate Limits

The free tier allows 500 requests/month. The scraper makes one request per invocation.

---

## Wikipedia: Historical World Cup Brackets

- **Source**: Wikipedia pages for each World Cup (e.g., "2022_FIFA_World_Cup")
- **Format**: Wikitext (parsed via MediaWiki API)
- **Access**: `src/scraping/scrape_historical_world_cups.py` -- `scrape_historical_brackets()`
- **Output**: `data/raw/historical_world_cups.csv`
- **Coverage**: 1930-2022 (excluding 1942, 1946)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | World Cup year |
| `round` | string | Round type ("group", "quarter finals", "semi finals", "final") |
| `home_team` | string | Home team name (normalized) |
| `away_team` | string | Away team name |
| `home_score` | int | Home team goals |
| `away_score` | int | Away team goals |

### Fallback

If Wikipedia scraping fails for any year, the scraper falls back to filtering the Kaggle international match results for `tournament == "FIFA World Cup"`.

---

## Live Results (Auto-Scraped)

- **Source**: ESPN (primary), Wikipedia (fallback — uses match-result-box parser for reliable scraping)
- **Access**: `src/scraping/scrape_live_results.py` -- `scrape_live_results()`
- **Output**: `data/raw/wc2026_results_live.csv`

Note: The Wikipedia scraper was rewritten to use match-result-box parsing instead of group-standings parsing, which correctly finds all 13 completed WC2026 matches (was only finding 10). The ESPN scraper filters out invalid rows (team names starting with "v", rows without scores).

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | string/datetime | Match date (may be null from Wikipedia) |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `home_score` | int/null | Home team goals (null if not yet played) |
| `away_score` | int/null | Away team goals (null if not yet played) |
| `source` | string | Data source ("espn" or "wikipedia") |

### Scrape Priority

1. ESPN is tried first
2. If ESPN returns no results, Wikipedia wikitext is parsed for score patterns
3. Manual override CSV is merged (manual takes priority for same match)

---

## Manual Override CSV

- **File**: `data/raw/wc2026_results_manual.csv`
- **Format**: CSV
- **Purpose**: Manually enter or correct WC2026 match results

### Template

```csv
date,home_team,away_team,home_score,away_score,group,match_number
```

### Example

```csv
date,home_team,away_team,home_score,away_score,group,match_number
2026-06-11,Mexico,New Zealand,3,0,A,1
2026-06-11,United States,Wales,2,1,B,2
```

### Usage

- Create the file at `data/raw/wc2026_results_manual.csv`
- Add rows as matches are played
- Run `python run_pipeline.py --step live-validate` to validate predictions
- Run `python run_pipeline.py --all --retrain` to retrain models with live data
- Manual entries take priority over auto-scraped results for the same match

---

## Confederation Mapping

- **File**: `data/external/continents.csv`
- **Format**: CSV (static, maintained in the repository)
- **Rows**: 211 countries

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `country` | string | Country name |
| `confederation` | string | Confederation (UEFA, CONMEBOL, CONCACAF, AFC, CAF, OFC) |
| `fifa_code` | string | 3-letter FIFA country code |

### Confederation Values

| Confederation | Region |
|---------------|--------|
| UEFA | Europe |
| CONMEBOL | South America |
| CONCACAF | North/Central America & Caribbean |
| AFC | Asia |
| CAF | Africa |
| OFC | Oceania |

---

## Data Freshness and Update Schedule

| Data Source | Recommended Update Frequency | How to Update |
|-------------|------------------------------|---------------|
| Kaggle match results | Monthly or before re-training | Delete `data/raw/international_matches/` and re-run scraping |
| Kaggle FIFA rankings | Monthly or before re-training | Delete `data/raw/fifa_rankings/` and re-run scraping |
| Current FIFA rankings | Before each re-training | Re-run `--step scraping` (always re-scrapes) |
| WC2026 groups | After final draw or qualifying updates | Delete `data/raw/wc2026_groups.csv` and re-run scraping |
| WC2026 fixtures | After schedule changes | Delete `data/raw/wc2026_fixtures.csv` and re-run scraping |
| Outright odds | Daily during tournament | Re-run `--step scraping` (requires `ODDS_API_KEY`) |
| Match odds | Before simulation | Re-run `--step scraping` |
| Live results | During tournament | Re-run `--step scraping` or `--step live-validate` |
| Manual overrides | As needed | Edit `data/raw/wc2026_results_manual.csv` |

### Forcing Re-Download

To force re-download of Kaggle data, delete the target directory:

```bash
rm -rf data/raw/international_matches/
rm -rf data/raw/fifa_rankings/
python run_pipeline.py --step scraping
```

Web-scraped data (Wikipedia, ESPN) is always re-scraped on each run.