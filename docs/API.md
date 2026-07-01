# API / Module Reference

This document provides a comprehensive reference for every public function, class, and constant in the World Cup 2026 ML Predictor.

---

## Table of Contents

- [Configuration (`src/config.py`)](#configuration-srcconfigpy)
- [Helpers (`src/helpers.py`)](#helpers-srchelperspy)
- [Scraping (`src/scraping/`)](#scraping-srcscraping)
- [Features (`src/features/`)](#features-srcfeatures)
- [Models (`src/models/`)](#models-srcmodels)
- [Simulation (`src/simulation/`)](#simulation-srcsimulation)
- [Visualization (`src/visualization/`)](#visualization-srcvisualization)
- [CLI (`run_pipeline.py`)](#cli-run_pipelinepy)

---

## Configuration (`src/config.py`)

All configuration constants are module-level variables in `src/config.py`.

### Paths

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `BASE_DIR` | `Path` | Project root | Base directory |
| `DATA_DIR` | `Path` | `BASE_DIR/data` | Data directory |
| `RAW_DIR` | `Path` | `DATA_DIR/raw` | Raw scraped data |
| `PROCESSED_DIR` | `Path` | `DATA_DIR/processed` | Processed features and models |
| `EXTERNAL_DIR` | `Path` | `DATA_DIR/external` | External static data |
| `MODELS_DIR` | `Path` | `DATA_DIR/processed` | Model storage (alias) |
| `FIGURES_DIR` | `Path` | `DATA_DIR/processed/evaluation` | Evaluation outputs |

### Random State and Simulation

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `RANDOM_STATE` | `int` | 42 | Random seed for reproducibility |
| `N_SIMULATIONS` | `int` | 1000 | Number of Monte Carlo tournament simulations |

### Kaggle Datasets

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `KAGGLE_MATCHES_DATASET` | `str` | `"martj42/international-football-results-from-1872-to-2017"` | Kaggle dataset slug for match results |
| `KAGGLE_RANKINGS_DATASET` | `str` | `"cashncarry/fifaworldranking"` | Kaggle dataset slug for FIFA rankings |

### Wikipedia URLs

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `WIKIPEDIA_API_BASE` | `str` | `"https://en.wikipedia.org/w/api.php"` | MediaWiki API endpoint |
| `WC2026_WIKI_PAGE` | `str` | `"2026_FIFA_World_Cup"` | Wikipedia page title for WC2026 |
| `FIFA_RANKINGS_WIKI_PAGE` | `str` | `"FIFA_Men%27s_World_Ranking"` | Wikipedia page title for rankings |

### Odds API

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `ODDS_API_BASE` | `str` | `"https://api.the-odds-api.com/v4"` | the-odds-api base URL |
| `ODDS_API_KEY` | `str` | `os.environ.get("ODDS_API_KEY", "")` | API key from environment |

### Tournament Format

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `HOST_NATIONS` | `list[str]` | `["United States", "Canada", "Mexico"]` | 2026 host countries |
| `N_GROUPS` | `int` | 12 | Number of groups |
| `GROUP_LETTERS` | `list[str]` | `["A", "B", ..., "L"]` | Group labels |
| `TEAMS_PER_GROUP` | `int` | 4 | Teams per group |
| `ADVANCE_PER_GROUP` | `int` | 2 | Top teams advancing per group |
| `BEST_THIRD_ADVANCE` | `int` | 8 | Best third-place teams advancing |

### Elo Parameters

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `ELO_HOME_ADVANTAGE` | `int` | 100 | Elo bonus for home team |
| `ELO_INITIAL_RATING` | `int` | 1000 | Starting Elo for new teams |
| `ELO_DRAW_FACTOR` | `float` | 0.30 | Draw probability scaling factor (increased from 0.25) |
| `K_FACTORS` | `dict` | `{"FIFA World Cup": 80, ...}` | K-factor by tournament type |

### Draw Calibration

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `WC_GROUP_DRAW_RATE` | `float` | 0.25 | Historical WC group-stage draw rate, used to calibrate under-predicted draws |

### Neural Network

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `NEURAL_NET_EPOCHS` | `int` | 100 | Max training iterations (overridden to 300 in `_build_sklearn_mlp`) |
| `NEURAL_NET_PATIENCE` | `int` | 10 | Early stopping patience |
| `NEURAL_NET_LAYERS` | `list[int]` | `[128, 64, 32]` | Hidden layer sizes |
| `NEURAL_NET_DROPOUT` | `float` | 0.3 | Not used (sklearn MLP uses alpha); import removed from train.py |
| `NEURAL_NET_LEARNING_RATE` | `float` | 1e-3 | Initial learning rate |

### Hyperparameter Tuning

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `OPTUNA_TRIALS` | `int` | 20 | Number of Optuna hyperparameter trials |
| `CV_FOLDS` | `int` | 3 | Cross-validation folds |

### Outcome Encoding

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `OUTCOME_HOME_WIN` | `int` | 1 | Home win label |
| `OUTCOME_DRAW` | `int` | 0 | Draw label |
| `OUTCOME_AWAY_WIN` | `int` | -1 | Away win label |
| `OUTCOME_LABELS` | `dict` | `{1: "home_win", 0: "draw", -1: "away_win"}` | Label mapping |
| `TEAM_NAME_MAPPING` | `dict` | 20+ entries | Country name aliases |
| `CONFEDERATIONS` | `dict` | 80+ entries | Country to confederation mapping |

---

## Helpers (`src/helpers.py`)

### `logger`

```python
logger: logging.Logger
```

Module-level logger with name `"worldcup"`, configured with streaming handler and format `%(asctime)s [%(levelname)s] %(name)s: %(message)s`.

### `ensure_dirs()`

```python
def ensure_dirs() -> None
```

Creates all required data directories (`DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `EXTERNAL_DIR`, `FIGURES_DIR`) if they do not exist. Logs each directory created.

### `setup_kaggle_credentials()`

```python
def setup_kaggle_credentials() -> None
```

Checks for `~/.kaggle/kaggle.json`. If not found, prompts the user for Kaggle username and API key interactively. Saves credentials and sets file permissions to 600. Raises `SystemExit` if credentials are not provided.

### `normalize_team_name(name: str) -> str`

```python
def normalize_team_name(name: str) -> str
```

Normalizes a team name by stripping whitespace and applying the `TEAM_NAME_MAPPING` dictionary. Returns the original name (stripped) if no mapping exists.

**Parameters:**
- `name` -- Raw team name string

**Returns:** Normalized team name

**Example:**
```python
normalize_team_name("USA")        # "United States"
normalize_team_name("Korea Republic")  # "South Korea"
normalize_team_name("  Brazil  ")  # "Brazil"
```

### `load_cached_data(path: Path, force_refresh: bool = False) -> pd.DataFrame | None`

```python
def load_cached_data(path: Path, force_refresh: bool = False) -> pd.DataFrame | None
```

Loads data from a cached file (Parquet or CSV). Returns `None` if the file does not exist or `force_refresh` is True.

**Parameters:**
- `path` -- Path to the data file
- `force_refresh` -- If True, skip cache and return None

**Returns:** DataFrame if file exists and `force_refresh` is False; otherwise None

---

## Scraping (`src/scraping/`)

### `download_kaggle.py`

#### `download_match_results(force: bool = False) -> Path`

```python
def download_match_results(force: bool = False) -> Path
```

Downloads the `martj42/international-football-results-from-1872-to-2017` dataset from Kaggle. Extracts to `data/raw/international_matches/`. Skips if all expected files (`results.csv`, `shootouts.csv`, `former_names.csv`) already exist and `force` is False.

**Parameters:**
- `force` -- Force re-download even if files exist

**Returns:** Path to the output directory

#### `download_fifa_rankings(force: bool = False) -> Path`

```python
def download_fifa_rankings(force: bool = False) -> Path
```

Downloads the `cashncarry/fifaworldranking` dataset from Kaggle. Extracts to `data/raw/fifa_rankings/`. Skips if `fifa_ranking.csv` already exists and `force` is False.

**Parameters:**
- `force` -- Force re-download

**Returns:** Path to the output directory

---

### `scrape_fifa_rankings.py`

#### `scrape_current_fifa_rankings() -> pd.DataFrame`

```python
def scrape_current_fifa_rankings() -> pd.DataFrame
```

Scrapes current FIFA men's world rankings from Wikipedia. Parses HTML tables for rank, country, and total points. Falls back to MediaWiki API wikitext parsing if HTML parsing fails. Saves to `data/raw/fifa_rankings_current.csv`.

**Returns:** DataFrame with columns `rank`, `country`, `total_points`

#### `merge_rankings(historical_path: Path, current_df: pd.DataFrame) -> pd.DataFrame`

```python
def merge_rankings(historical_path: Path, current_df: pd.DataFrame) -> pd.DataFrame
```

Merges historical FIFA rankings (Kaggle) with current rankings (Wikipedia). Appends current data with today's date, propagates confederation information, and saves to `data/raw/fifa_rankings_merged.csv`.

**Parameters:**
- `historical_path` -- Path to historical rankings CSV
- `current_df` -- Current rankings DataFrame from Wikipedia

**Returns:** Merged DataFrame

---

### `scrape_world_cup_2026.py`

#### `scrape_wc2026_groups() -> pd.DataFrame`

```python
def scrape_wc2026_groups() -> pd.DataFrame
```

Scrapes WC2026 group compositions from Wikipedia. First attempts wikitext parsing via the MediaWiki API, then falls back to HTML parsing. Normalizes team names. Saves to `data/raw/wc2026_groups.csv`.

**Returns:** DataFrame with columns `group`, `team`, `pot`

#### `scrape_wc2026_fixtures() -> pd.DataFrame`

```python
def scrape_wc2026_fixtures() -> pd.DataFrame
```

Scrapes WC2026 match fixtures from Wikipedia. Attempts wikitext parsing then HTML fallback. Maps teams to groups using the groups CSV. Uses `combinations` (not permutations) for correct fixture generation (72 matches, not 144). Saves to `data/raw/wc2026_fixtures.csv`.

**Returns:** DataFrame with columns `match_number`, `date`, `home_team`, `away_team`, `group`, `venue`, `city`

---

### `scrape_odds.py`

#### `odds_to_probability(american_odds: int) -> float`

```python
def odds_to_probability(american_odds: int) -> float
```

Converts American odds to implied probability.

**Parameters:**
- `american_odds` -- American odds value (positive = underdog, negative = favorite)

**Returns:** Implied probability between 0 and 1

**Example:**
```python
odds_to_probability(200)    # 0.3333 (underdog)
odds_to_probability(-200)   # 0.6667 (favorite)
```

#### `fetch_outright_odds(api_key: str | None = None) -> pd.DataFrame`

```python
def fetch_outright_odds(api_key: str | None = None) -> pd.DataFrame
```

Fetches outright World Cup winner odds from the-odds-api. Requires `ODDS_API_KEY` environment variable or `api_key` parameter. Saves raw JSON and processed CSV.

**Parameters:**
- `api_key` -- Optional API key (overrides env var)

**Returns:** DataFrame with columns `team`, `american_odds`, `implied_probability`, `bookmaker`, `bookmaker_key`, `last_update`. Empty DataFrame if no API key.

#### `scrape_match_odds() -> pd.DataFrame`

```python
def scrape_match_odds() -> pd.DataFrame
```

Scrapes per-match odds from ESPN for WC2026 fixtures. Iterates over all fixtures and attempts to find matching odds on the ESPN schedule page. Saves to `data/raw/odds_match.csv`.

**Returns:** DataFrame with columns `home_team`, `away_team`, `home_american_odds`, `draw_american_odds`, `away_american_odds`, `home_implied_prob`, `draw_implied_prob`, `away_implied_prob`

---

### `scrape_live_results.py`

#### `scrape_live_results() -> pd.DataFrame`

```python
def scrape_live_results() -> pd.DataFrame
```

Scrapes live WC2026 match results. Tries ESPN first, then Wikipedia. Merges with manual overrides from `data/raw/wc2026_results_manual.csv` (manual entries take priority). Saves to `data/raw/wc2026_results_live.csv`.

**Returns:** DataFrame with columns `date`, `home_team`, `away_team`, `home_score`, `away_score`, `source`

---

### `scrape_squad_quality.py`

#### `scrape_squad_quality(force: bool = False) -> pd.DataFrame`

```python
def scrape_squad_quality(force: bool = False) -> pd.DataFrame
```

Scrapes Transfermarkt for squad quality data for all 48 WC 2026 teams. Uses German-slug URL construction with `TEAM_SLUG_OVERRIDES` for teams whose names don't match Transfermarkt's URL format, and `TEAM_SEARCH_OVERRIDES` for teams that must be found via search. Rate-limited at 2 seconds between requests.

**Parameters:**
- `force` -- Force re-scrape even if file exists

**Returns:** DataFrame with columns `team`, `squad_market_value_m`, `squad_size`, `avg_age`, `foreigners`, `avg_player_value_m`, `top_player_value_m`

**Data highlights:** Top: France (€1.52B), England (€1.36B), Spain (€1.22B); Mid: USA (€386M), Morocco (€448M), Japan (€271M); Bottom: Qatar (€20M), Jordan (€20M), Iraq (€21M)

---

### `scrape_historical_world_cups.py`

#### `scrape_historical_brackets() -> pd.DataFrame`

```python
def scrape_historical_brackets() -> pd.DataFrame
```

Scrapes historical World Cup brackets (1930-2022, excluding 1942 and 1946) from Wikipedia. Falls back to filtering Kaggle data if scraping fails. Saves to `data/raw/historical_world_cups.csv`.

**Returns:** DataFrame with columns `year`, `round`, `home_team`, `away_team`, `home_score`, `away_score`

---

## Features (`src/features/`)

### `elo.py` -- EloRatingSystem

```python
class EloRatingSystem:
    def __init__(
        self,
        k_factor: int = 40,
        home_advantage: int = ELO_HOME_ADVANTAGE,  # 100
        initial_rating: float = ELO_INITIAL_RATING,  # 1000
        draw_factor: float = ELO_DRAW_FACTOR,  # 0.30
    )
```

Elo rating system for international football. Uses `ELO_DRAW_FACTOR=0.30` for improved draw probability calibration.

**Parameters:**
- `k_factor` -- Default K-factor for rating updates (overridden by tournament-specific K-factors)
- `home_advantage` -- Elo bonus added to home team rating (default: 100)
- `initial_rating` -- Starting rating for new teams (default: 1000)
- `draw_factor` -- Scaling factor for draw probability (default: 0.30)

**Attributes:**
- `ratings` -- `dict[str, float]`: Team name to current Elo rating
- `rating_history` -- `list[dict]`: History of rating changes per match

#### `compute_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame`

```python
def compute_elo_ratings(self, matches_df: pd.DataFrame) -> pd.DataFrame
```

Computes Elo ratings from a DataFrame of matches. Processes matches chronologically, applying tournament-specific K-factors and home advantage. Saves current ratings to `data/processed/elo_ratings_current.parquet` and history to `data/processed/elo_ratings_history.parquet`.

**Parameters:**
- `matches_df` -- DataFrame with columns `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `neutral`

**Returns:** DataFrame with columns `team`, `elo` (sorted by rating descending)

#### `predict_match_probability(home_elo: float, away_elo: float, neutral: bool = True) -> dict[str, float]`

```python
def predict_match_probability(
    self, home_elo: float, away_elo: float, neutral: bool = True
) -> dict[str, float]
```

Predicts match outcome probabilities from Elo ratings. Uses `ELO_DRAW_FACTOR=0.30` for draw probability scaling.

**Parameters:**
- `home_elo` -- Home team Elo rating
- `away_elo` -- Away team Elo rating
- `neutral` -- If True, no home advantage bonus

**Returns:** Dict with keys `"home_win"`, `"draw"`, `"away_win"` and probability values

**Example:**
```python
elo = EloRatingSystem()
elo.compute_elo_ratings(matches_df)
probs = elo.predict_match_probability(
    elo.get_team_rating("Brazil"),
    elo.get_team_rating("Argentina"),
    neutral=True,
)
# {"home_win": 0.42, "draw": 0.28, "away_win": 0.30}
```

#### `get_team_rating(team: str) -> float`

```python
def get_team_rating(self, team: str) -> float
```

Returns the current Elo rating for a team. Returns `initial_rating` if the team is unknown.

#### `get_top_ratings(n: int = 30) -> pd.DataFrame`

```python
def get_top_ratings(self, n: int = 30) -> pd.DataFrame
```

Returns the top N teams by Elo rating.

**Parameters:**
- `n` -- Number of top teams to return

**Returns:** DataFrame with columns `team`, `elo`

---

### `build_features.py`

#### `FEATURES_CACHE_VERSION = "7"`

Feature cache version. Incremented when feature definitions change, invalidating all stale caches. Updated from "6" to "7" to add squad quality features.

#### `load_squad_quality() -> pd.DataFrame`

```python
def load_squad_quality() -> pd.DataFrame
```

Loads the squad quality data from `data/raw/squad_quality.csv`. Returns an empty DataFrame with expected columns if the file does not exist.

**Returns:** DataFrame with columns `team`, `squad_market_value_m`, `squad_size`, `avg_age`, `foreigners`, `avg_player_value_m`, `top_player_value_m`

#### `get_team_squad_value(team: str, squad_df: pd.DataFrame) -> dict`

```python
def get_team_squad_value(team: str, squad_df: pd.DataFrame) -> dict
```

Looks up squad quality features for a given team from the squad quality DataFrame. Returns confederation-average fallbacks if the team is not found.

**Parameters:**
- `team` -- Team name (normalized)
- `squad_df` -- Squad quality DataFrame from `load_squad_quality()`

**Returns:** Dict with keys `squad_market_value_m`, `avg_player_value_m`, `top_player_value_m` (or confederation-average fallbacks)

#### `load_all_data(include_live: bool = False) -> dict[str, pd.DataFrame]`

```python
def load_all_data(include_live: bool = False) -> dict[str, pd.DataFrame]
```

Loads all raw data sources into a dictionary of DataFrames. Normalizes team names and parses dates.

**Parameters:**
- `include_live` -- Whether to include live WC2026 results

**Returns:** Dict with keys `"matches"`, `"shootouts"`, `"rankings"`, `"current_rankings"`, `"continents"`, and optionally `"live_results"`, `"wc_history"`

#### `build_match_features(matches_df: pd.DataFrame | None = None, elo_system: EloRatingSystem | None = None, include_live: bool = False) -> pd.DataFrame`

```python
def build_match_features(
    matches_df: pd.DataFrame | None = None,
    elo_system: EloRatingSystem | None = None,
    include_live: bool = False,
) -> pd.DataFrame
```

Builds feature vectors for every match. Computes Elo ratings, FIFA rankings, form (last 5/10), EWM form, head-to-head, strength of schedule, confederation, tournament type, draw-predictive features, interaction features, and squad quality features (with confederation-average fallbacks for historical matches). Results are cached with version "7" -- cache is rebuilt if input files change. Squad quality CSV is included in cache hash computation. Saves to `data/processed/match_features.parquet`.

**Parameters:**
- `matches_df` -- Match data (loaded from Kaggle if None)
- `elo_system` -- Pre-computed Elo system (computed if None)
- `include_live` -- Include live WC2026 results

**Returns:** DataFrame with 80 feature columns per match

#### Helper Functions

```python
def _get_fifa_ranking(team: str, date: pd.Timestamp, rankings_df: pd.DataFrame) -> dict
```
Returns `{"fifa_rank": float, "fifa_points": float}` for a team at a given date.

```python
def _get_fifa_ranking_fast(team: str, date: pd.Timestamp, rankings_by_team: dict[str, pd.DataFrame]) -> dict
```
Optimized version using precomputed per-team rankings DataFrames.

```python
def _get_confederation(team: str, continents_df: pd.DataFrame | None = None) -> str
```
Returns the confederation string for a team.

```python
def _compute_form_features(team: str, date: pd.Timestamp, matches_df: pd.DataFrame, n: int = 10) -> dict
```
Returns last-N form features: `win_rate`, `draw_rate`, `loss_rate`, `goals_scored_avg`, `goals_conceded_avg`, `goal_diff_avg`, `clean_sheet_rate`.

```python
def _compute_h2h_features(home: str, away: str, date: pd.Timestamp, matches_df: pd.DataFrame, n: int = 5) -> dict
```
Returns head-to-head features: `h2h_home_wins`, `h2h_draws`, `h2h_away_wins`, `h2h_draw_rate`, `h2h_home_goals_avg`, `h2h_away_goals_avg`.

---

### `build_2026_features.py`

#### `build_wc2026_features(groups_df: pd.DataFrame | None = None, fixtures_df: pd.DataFrame | None = None, elo_system: EloRatingSystem | None = None, rankings_df: pd.DataFrame | None = None, odds_df: pd.DataFrame | None = None, include_live: bool = True) -> pd.DataFrame`

```python
def build_wc2026_features(
    groups_df: pd.DataFrame | None = None,
    fixtures_df: pd.DataFrame | None = None,
    elo_system: EloRatingSystem | None = None,
    rankings_df: pd.DataFrame | None = None,
    odds_df: pd.DataFrame | None = None,
    include_live: bool = True,
) -> pd.DataFrame
```

Builds feature vectors for all WC2026 scheduled matches. Uses current Elo ratings (recomputed from full match history if cache has fewer than 10 teams), current FIFA rankings, form computed through June 2026, and odds if available. Handles neutral venue flags correctly: neutral venues get `neutral=1`, host nation home matches get `neutral=0` with `home_advantage=1.0`. Saves to `data/processed/wc2026_match_features.parquet`.

**Returns:** DataFrame with feature columns for each WC2026 match

---

## Models (`src/models/`)

### `train.py`

#### `FEATURE_COLUMNS` and `OPTIONAL_FEATURES`

Two lists defining the feature columns used by models:

- `FEATURE_COLUMNS` (27): Core features always present, including `elo_close`, `draw_tendency`, `fifa_close`
- `OPTIONAL_FEATURES` (53): Form, EWM form, H2H, SOS, draw, tournament, odds, interaction, and squad quality features (may be absent, filled with zeros)

Total: 80 features when all are present.

#### `_compute_sample_weights(y_train)`

```python
def _compute_sample_weights(y_train) -> np.ndarray
```

Computes sample weights for class imbalance. Draw class (label 1) receives 4.0x weight by default (used by XGBoost); NeuralNet overrides to 8.0x via `_compute_sample_weights(y_train, draw_weight=8.0)`. Home win (2) and away win (0) receive 1.0x. RF/LogReg use `class_weight="balanced"` instead of sample weights.

**Returns:** Array of sample weights aligned with `y_train`

#### `split_data(df: pd.DataFrame, test_year: int = 2023, val_years: list[int] | None = None) -> tuple`

```python
def split_data(
    df: pd.DataFrame,
    test_year: int = 2023,
    val_years: list[int] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame,
]
```

Splits match features into train/validation/test sets by year. Default: train < 2023, val = 2023, test > 2023. Maps outcomes to `{1: 2, 0: 1, -1: 0}` for scikit-learn. Applies `SimpleImputer(strategy="median")` to fill missing values and saves the fitted imputer to `data/processed/models/imputer.joblib` (with `compress=3`).

**Returns:** `(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train_df, val_df, test_df)`

#### `train_xgboost(X_train, y_train, X_val=None, y_val=None) -> dict`

```python
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict
```

Trains an XGBClassifier with Optuna hyperparameter optimization. Optimizes for **log_loss** (not accuracy). Uses 4x sample weight for draw class (default from `_compute_sample_weights`). Uses validation set for early stopping if provided; otherwise uses TimeSeriesSplit cross-validation. Optuna capped at 20 trials with 3-fold CV.

**Returns:** `{"model": XGBClassifier, "params": dict, "name": "XGBoost"}`

#### `train_random_forest(X_train, y_train) -> dict`

```python
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> dict
```

Trains a RandomForestClassifier with grid search over hyperparameters using 3-fold TimeSeriesSplit. Optimizes for `neg_log_loss` with `class_weight="balanced"`. Searches `n_estimators=[100,200]`, `max_depth=[10,20]` (capped at 20), `min_samples_split=[2,5]`.

**Returns:** `{"model": RandomForestClassifier, "params": dict, "name": "RandomForest"}`

#### `train_logistic_regression(X_train, y_train) -> dict`

```python
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> dict
```

Trains a LogisticRegression baseline with StandardScaler in a Pipeline. Uses `class_weight="balanced"` and GridSearchCV over `C=[0.01, 0.1, 1.0, 10.0]` with 3-fold TimeSeriesSplit, optimizing for `neg_log_loss`.

**Returns:** `{"model": Pipeline, "params": dict, "name": "LogisticRegression"}`

#### `train_lightgbm(X_train, y_train, X_val=None, y_val=None) -> dict`

```python
def train_lightgbm(X_train, y_train, X_val=None, y_val=None) -> dict
```

**Note:** This function is preserved in the codebase but is not called during training. LightGBM was removed from the ensemble due to poor performance (0.49 accuracy, 0.99 log_loss).

**Returns:** `{"model": LGBMClassifier, "params": dict, "name": "LightGBM"}`

#### `train_neural_net(X_train, y_train, X_val=None, y_val=None) -> dict`

```python
def train_neural_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict
```

Trains an sklearn MLPClassifier (3-layer: 128-64-32) with alpha=0.001, batch_size=256, adaptive learning rate, max_iter=300, and early stopping with patience=10. Uses 8x sample weight for draw class. If validation data is provided, combines train+val for final fit.

**Returns:** `{"model": MLPClassifier, "params": dict, "name": "NeuralNet"}`

#### `save_all_models(models: dict[str, dict], feature_cols: list[str]) -> None`

```python
def save_all_models(models: dict[str, dict], feature_cols: list[str]) -> None
```

Saves all models to `data/processed/models/` as Joblib files (with `compress=3`) and feature columns as `feature_columns.joblib`.

---

### `ensemble.py`

#### `CalibratedWrapper`

```python
class CalibratedWrapper:
    def __init__(self, base_model, isotonic_regressors=None, n_classes=3)
```

**Note:** This class is preserved for future experimentation but is not currently used. Isotonic calibration was tested and removed because it degraded test performance (log_loss 0.8374 -> 1.0436).

Wraps a base model with per-class isotonic calibration. Provides `predict_proba()` and `predict()` methods that apply calibration then normalize probabilities.

#### `WeightedEnsemble`

```python
class WeightedEnsemble:
    def __init__(self, sklearn_model, nn_model, X_val=None, y_val=None)
```

A weighted ensemble that blends predictions from a scikit-learn model and a neural network model. Optimizes the blending weight on validation data.

#### `build_stacking_ensemble(models: dict[str, dict], X_train: np.ndarray, y_train: np.ndarray) -> dict`

```python
def build_stacking_ensemble(
    models: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict
```

Builds a StackingClassifier with LogisticRegression meta-learner (no class_weight). Only includes models that implement `predict_proba`. Uses `KFold(n_splits=3, shuffle=True)` for cross-validated stacking.

**Parameters:**
- `models` -- Dict of model dicts (each with `"model"` key)
- `X_train` -- Training features
- `y_train` -- Training labels

**Returns:** `{"model": StackingClassifier, "params": dict, "name": "StackingEnsemble"}`

#### `build_voting_ensemble(models: dict[str, dict], X_train: np.ndarray, y_train: np.ndarray, weights: list[float] | None = None) -> dict`

```python
def build_voting_ensemble(
    models: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: list[float] | None = None,
) -> dict
```

Builds a soft VotingClassifier with optional custom weights.

#### `calibrate_ensemble(ensemble_model, X_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic") -> dict`

```python
def calibrate_ensemble(
    ensemble_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> dict
```

**Note:** This function is preserved but not currently used. Calibration hurt test performance.

Calibrates ensemble probabilities using per-class `IsotonicRegression`. Supports `"isotonic"` and `"sigmoid"` methods.

#### `build_best_ensemble(models: dict[str, dict], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict`

```python
def build_best_ensemble(
    models: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict
```

Builds and selects the best ensemble by evaluating multiple candidates on validation data using **log_loss** as the selection metric (not accuracy):

1. Evaluates each individual base model
2. Builds a VotingEnsemble with uniform weights
3. Builds a WeightedVotingEnsemble with inverse-log-loss weights
4. Builds a StackingEnsemble with LogisticRegression meta-learner
5. Selects the candidate with the lowest validation log_loss
6. Saves the best model to `data/processed/models/best_model.joblib`

The current best ensemble is a WeightedVotingEnsemble (soft VotingClassifier with inverse-log-loss weights) using 4 base models (XGBoost, RandomForest, LogisticRegression, NeuralNet). Selected over VotingEnsemble (uniform weights) and StackingEnsemble by validation log_loss.

**Returns:** Dict with `"model"` and `"name"` keys

---

### `evaluate.py`

#### `evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict`

```python
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict
```

Evaluates a single model on test data. Computes accuracy, log loss, Brier scores per class, classification report, and confusion matrix.

**Returns:** Dict with keys `model_name`, `accuracy`, `log_loss`, `brier_scores`, `classification_report`, `confusion_matrix`, `y_pred`, `y_prob`

#### `compare_models(results_dict: dict[str, dict]) -> pd.DataFrame`

```python
def compare_models(results_dict: dict[str, dict]) -> pd.DataFrame
```

Creates a comparison DataFrame of all models sorted by log loss.

#### `generate_evaluation_report(all_results: dict[str, dict], feature_cols: list[str] | None = None) -> pd.DataFrame`

```python
def generate_evaluation_report(
    all_results: dict[str, dict],
    feature_cols: list[str] | None = None,
) -> pd.DataFrame
```

Generates a complete evaluation report: model comparison CSV, calibration curves PNG, feature importance PNG, and JSON summary. Saves to `data/processed/evaluation/`.

---

### `live_validation.py`

#### `validate_against_live(model, feature_cols: list[str]) -> dict`

```python
def validate_against_live(
    model,
    feature_cols: list[str],
) -> dict
```

Validates model predictions against played WC2026 matches. Loads live results, matches them to WC2026 features, predicts outcomes (with swapped fixture handling), and computes accuracy and log-loss.

**Returns:** Dict with keys `accuracy_argmax`, `accuracy_threshold`, `correct_argmax`, `correct_threshold`, `log_loss`, `total_matches`, `results` (DataFrame). Empty dict if no live data available.

The function returns both `accuracy_argmax` (standard argmax prediction accuracy) and `accuracy_threshold` (accuracy using draw-threshold prediction). The `results` DataFrame contains a `draw_ratio` column showing the ratio of draw probability to the max non-draw probability for each match.

---

### `prediction.py`

#### `predict_with_draw_threshold(model, X, draw_threshold=0.85) -> np.ndarray`

```python
def predict_with_draw_threshold(
    model,
    X: np.ndarray,
    draw_threshold: float = 0.85,
) -> np.ndarray
```

Predicts class labels with a draw-aware threshold. If the ratio of draw probability to the max non-draw probability exceeds `draw_threshold`, the prediction is overridden to draw (class 1).

**Parameters:**
- `model` -- Trained model with `predict_proba` method
- `X` -- Feature matrix
- `draw_threshold` -- Threshold for draw ratio (draw_prob / max(non_draw_probs)) above which draw is predicted (default: 0.85)

**Returns:** Array of predicted class labels

#### `predict_proba_with_draw_boost(model, X, draw_boost=1.0) -> np.ndarray`

```python
def predict_proba_with_draw_boost(
    model,
    X: np.ndarray,
    draw_boost: float = 1.0,
) -> np.ndarray
```

Predicts class probabilities with an optional boost to the draw class. When `draw_boost > 1.0`, the draw probability column is multiplied by the boost factor and probabilities are renormalized to sum to 1.

**Parameters:**
- `model` -- Trained model with `predict_proba` method
- `X` -- Feature matrix
- `draw_boost` -- Multiplicative boost for draw probability (default: 1.0, no boost)

**Returns:** Array of predicted probabilities (N x 3)

---

## Simulation (`src/simulation/`)

### Probability Handling

Both simulators handle `predict_proba()` output correctly:

- **Class ordering**: `predict_proba()` returns probabilities in class order `[away_win(0), draw(1), home_win(2)]`
- **Reordering**: Output is reordered to `[home_win, draw, away_win]` for simulation use
- **Swap handling**: When a fixture lookup falls back to the reversed match, a `swapped` flag is set and probabilities are swapped `[home_win, draw, away_win]` → `[away_win, draw, home_win]`
- **Imputation**: NaN features are filled using the trained `SimpleImputer` (median strategy) loaded from `data/processed/models/imputer.joblib`, not `np.nanmean()`
- **Normalization**: Probabilities are normalized (`probs / probs.sum()`) before sampling to prevent `ValueError: Probabilities do not sum to 1`

### Draw Calibration

During group stage simulation, predicted draw probabilities that fall below the historical WC group-stage draw rate (~25%) are boosted using `_calibrate_draw()`:

```python
WC_GROUP_DRAW_RATE = 0.25

def _calibrate_draw(probs):
    """Boost draw probability toward historical ~25% WC group-stage rate."""
    home_win_prob, draw_prob, away_win_prob = probs
    if draw_prob < WC_GROUP_DRAW_RATE:
        deficit = WC_GROUP_DRAW_RATE - draw_prob
        draw_prob = WC_GROUP_DRAW_RATE
        total_win_loss = home_win_prob + away_win_prob
        if total_win_loss > 0:
            scale = (1.0 - draw_prob) / total_win_loss
            home_win_prob *= scale
            away_win_prob *= scale
    return np.array([home_win_prob, draw_prob, away_win_prob])
```

Draw calibration is applied in `_predict_match()` (group_stage) and `_predict_match_proba()` (simulator), but NOT in knockout stage (where draws are redistributed to produce a winner).

### `group_stage.py` -- GroupStageSimulator

```python
class GroupStageSimulator:
    def __init__(
        self,
        model,
        feature_cols: list[str],
        groups_df: pd.DataFrame,
        n_simulations: int = 1000,
        imputer=None,
    )
```

**Parameters:**
- `model` -- Trained model with `predict_proba` method
- `feature_cols` -- List of feature column names
- `groups_df` -- DataFrame with columns `group`, `team`, `pot`
- `n_simulations` -- Number of Monte Carlo iterations
- `imputer` -- Optional fitted SimpleImputer (loaded from `imputer.joblib` if not provided)

#### `_predict_match(home: str, away: str, wc_features: pd.DataFrame) -> np.ndarray`

```python
def _predict_match(
    self, home: str, away: str, wc_features: pd.DataFrame
) -> np.ndarray
```

Predicts match outcome probabilities. Loads feature row, imputes NaN values using the trained imputer, calls `predict_proba()`, normalizes, and reorders from `[away_win, draw, home_win]` to `[home_win, draw, away_win]`. Handles swapped fixture lookups (when the match is found as away vs home). Applies draw calibration using `_calibrate_draw()` to boost draw probability toward the historical ~25% WC group-stage rate.

**Returns:** `np.array([home_win_prob, draw_prob, away_win_prob])`

#### `simulate_match(home: str, away: str, wc_features: pd.DataFrame) -> tuple[str, int, int]`

```python
def simulate_match(
    self, home: str, away: str, wc_features: pd.DataFrame
) -> tuple[str, int, int]
```

Simulates a single match. Returns `(outcome, home_goals, away_goals)` where outcome is `"home_win"`, `"draw"`, or `"away_win"`. Probabilities are normalized before sampling.

#### `simulate_group(group_letter: str, wc_features: pd.DataFrame) -> pd.DataFrame`

```python
def simulate_group(
    self, group_letter: str, wc_features: pd.DataFrame
) -> pd.DataFrame
```

Simulates all matches in one group and returns standings sorted by points, goal difference, goals scored.

#### `determine_third_place_qualifiers(all_group_results: list[pd.DataFrame]) -> list[str]`

```python
def determine_third_place_qualifiers(
    self, all_group_results: list[pd.DataFrame]
) -> list[str]
```

Ranks all third-place teams and returns the top 8 that qualify for the knockout stage.

#### `simulate_all_groups(wc_features: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]`

```python
def simulate_all_groups(
    self, wc_features: pd.DataFrame
) -> tuple[list[pd.DataFrame], list[str]]
```

Simulates all 12 groups and returns group standings and third-place qualifiers.

#### `get_advancement_probabilities(wc_features: pd.DataFrame) -> pd.DataFrame`

```python
def get_advancement_probabilities(
    self, wc_features: pd.DataFrame
) -> pd.DataFrame
```

Runs `n_simulations` group stage iterations and computes advancement probabilities. Saves to `data/processed/group_stage_probabilities.csv`.

**Returns:** DataFrame with columns `team`, `group`, `prob_1st`, `prob_2nd`, `prob_3rd`, `prob_4th`, `prob_advance`

---

### `knockout_stage.py` -- KnockoutStageSimulator

```python
class KnockoutStageSimulator:
    def __init__(
        self,
        model,
        feature_cols: list[str],
        n_simulations: int = 1000,
        imputer=None,
    )
```

**Parameters:**
- `model` -- Trained model with `predict_proba` method
- `feature_cols` -- List of feature column names
- `n_simulations` -- Number of Monte Carlo iterations
- `imputer` -- Optional fitted SimpleImputer (loaded from `imputer.joblib` if not provided)

#### `predict_knockout_match(home: str, away: str, wc_features: pd.DataFrame) -> np.ndarray`

```python
def predict_knockout_match(
    self, home: str, away: str, wc_features: pd.DataFrame
) -> np.ndarray
```

Predicts knockout match probabilities. Imputes NaN features, calls `predict_proba()`, normalizes, and reorders from `[away_win, draw, home_win]` to `[home_win, draw, away_win]`. Handles swapped fixture lookups. Does NOT apply draw calibration (draws are redistributed in knockout).

**Returns:** `np.array([home_win_prob, draw_prob, away_win_prob])`

#### `simulate_knockout_match(home: str, away: str, wc_features: pd.DataFrame) -> str`

```python
def simulate_knockout_match(
    self, home: str, away: str, wc_features: pd.DataFrame
) -> str
```

Simulates a knockout match. Draw probability is redistributed 55/45 favoring the stronger team (70% draw reduction). Returns the winning team name.

The redistribution uses the corrected probability ordering: `probs[0]` = home_win, `probs[1]` = draw, `probs[2]` = away_win.

#### `build_round_of_32_bracket(group_results: list[pd.DataFrame], third_place_teams: list[str]) -> dict`

```python
def build_round_of_32_bracket(
    self,
    group_results: list[pd.DataFrame],
    third_place_teams: list[str],
) -> dict
```

Constructs the Round of 32 bracket from group standings and third-place qualifiers.

#### `simulate_full_knockout(group_results: list[pd.DataFrame], third_place_teams: list[str], wc_features: pd.DataFrame) -> dict`

```python
def simulate_full_knockout(
    self,
    group_results: list[pd.DataFrame],
    third_place_teams: list[str],
    wc_features: pd.DataFrame,
) -> dict
```

Simulates the complete knockout bracket (R32 through Final).

**Returns:** Dict with keys `"ro32"`, `"ro16"`, `"qf"`, `"sf"`, `"final"` (string team name)

---

### `simulator.py` -- WorldCupSimulator

```python
class WorldCupSimulator:
    def __init__(
        self,
        model=None,
        feature_cols: list[str] | None = None,
        n_simulations: int = 1000,
        model_name: str = "xgboost",
    )
```

Orchestrates the full tournament simulation. By default, loads the XGBoost model (`xgboost.joblib`) for speed (819KB compressed). Falls back to `best_model.joblib` if XGBoost is not available.

**Parameters:**
- `model` -- Pre-loaded model (overrides file loading)
- `feature_cols` -- Pre-loaded feature columns (overrides file loading)
- `n_simulations` -- Number of Monte Carlo iterations
- `model_name` -- Name of the model file to load (default: "xgboost")

#### `run_full_simulation() -> pd.DataFrame`

```python
def run_full_simulation(self) -> pd.DataFrame
```

Runs `n_simulations` complete tournaments. For each iteration: simulates groups, applies live results if available, simulates knockout bracket, and tracks team advancement. Saves to `data/processed/tournament_probabilities.csv`.

**Returns:** DataFrame with columns `team`, `prob_ro32`, `prob_ro16`, `prob_qf`, `prob_sf`, `prob_final`, `prob_winner`

#### `get_results() -> pd.DataFrame`

Returns cached results from `data/processed/tournament_probabilities.csv`.

#### `save_results(results: pd.DataFrame = None) -> None`

Saves results to CSV.

### `run_simulation(n_simulations: int = None, model=None, feature_cols: list[str] | None = None, model_name: str = "xgboost") -> pd.DataFrame`

```python
def run_simulation(
    n_simulations: int | None = None,
    model=None,
    feature_cols: list[str] | None = None,
    model_name: str = "xgboost",
) -> pd.DataFrame
```

Convenience function to create a `WorldCupSimulator` and run a full simulation. Defaults to XGBoost model for speed.

---

## Visualization (`src/visualization/`)

### `plots.py`

#### `plot_tournament_probabilities(df: pd.DataFrame, top_n: int = 20, save: bool = True) -> Figure`

```python
def plot_tournament_probabilities(
    df: pd.DataFrame, top_n: int = 20, save: bool = True
) -> matplotlib.figure.Figure
```

Creates a horizontal bar chart of tournament winning probabilities for the top N teams.

#### `plot_group_heatmaps(group_probs: pd.DataFrame, save: bool = True) -> Figure`

Creates a heatmap grid showing advancement probabilities per position for each group.

#### `plot_feature_importance(model, feature_cols: list[str], top_n: int = 20, save: bool = True) -> Figure | None`

Plots top N feature importances for tree-based models.

#### `plot_elo_ratings(ratings_df: pd.DataFrame, top_n: int = 30, save: bool = True) -> Figure`

Creates a horizontal bar chart of top N Elo ratings.

#### `plot_model_comparison(comparison_df: pd.DataFrame, save: bool = True) -> Figure`

Creates a side-by-side bar chart comparing model accuracy and log loss.

#### `plot_round_probabilities(df: pd.DataFrame, save: bool = True) -> Figure`

Creates a grouped bar chart showing round advancement probabilities for the top 16 teams.

---

### `tables.py`

#### `format_power_rankings(df: pd.DataFrame, top_n: int = 30) -> str`

```python
def format_power_rankings(df: pd.DataFrame, top_n: int = 30) -> str
```

Formats a text table of power rankings with win%, final%, SF%, QF%, Ro16%.

#### `format_group_tables(group_probs: pd.DataFrame) -> str`

```python
def format_group_tables(group_probs: pd.DataFrame) -> str
```

Formats per-group probability tables showing 1st%, 2nd%, 3rd%, 4th%, advance%.

#### `format_bracket_summary(df: pd.DataFrame, top_n: int = 16) -> str`

```python
def format_bracket_summary(df: pd.DataFrame, top_n: int = 16) -> str
```

Formats a bracket summary table showing team advancement through each round.

---

## CLI (`run_pipeline.py`)

### `main()`

```python
def main() -> None
```

Entry point for the CLI. Parses arguments and dispatches to pipeline stages.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--step` | `choice` | None | Run specific step: `scraping`, `features`, `train`, `ensemble`, `evaluate`, `simulate`, `visualize`, `live-validate` |
| `--all` | `flag` | False | Run the complete pipeline |
| `--retrain` | `flag` | False | Include live WC2026 data in training |
| `--live-validate` | `flag` | False | Validate against live results |
| `--n-simulations` | `int` | 1000 | Override number of Monte Carlo simulations |
| `--setup-only` | `flag` | False | Only create directories and install dependencies |
| `--debug` | `flag` | False | Enable debug logging |

### Stage Functions

Each stage function is called by `main()` based on CLI arguments:

```python
def run_scraping() -> None       # Download data, scrape Wikipedia/ESPN/API
def run_features(include_live: bool = False) -> None  # Build Elo ratings and features
def run_train(include_live: bool = False) -> None     # Train all models
def run_ensemble() -> None       # Build best ensemble (by log_loss)
def run_evaluate() -> None       # Evaluate and compare models
def run_live_validate() -> None  # Validate against live results
def run_simulate() -> None       # Run Monte Carlo simulation (XGBoost model)
def run_visualize() -> None      # Generate plots and tables
```