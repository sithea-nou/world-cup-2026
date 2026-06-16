# World Cup 2026 ML Predictor

An end-to-end machine learning pipeline for predicting FIFA World Cup 2026 outcomes. The system combines 48,000+ international match results, FIFA rankings, Transfermarkt squad quality data, bookmaker odds, and Elo ratings to train an ensemble classifier (selected by validation log_loss from Voting, WeightedVoting, and Stacking candidates), then runs Monte Carlo simulations (1,000 iterations by default) using XGBoost for speed to estimate tournament advancement and winning probabilities for all 48 teams.

## Architecture

```
run_pipeline.py --all --n-simulations 1000
  Step 1: Data Scraping (Kaggle + Wikipedia + ESPN odds)
  Step 2: Feature Engineering (Elo, FIFA, form, H2H, SOS, EWM, draw features, squad quality)
  Step 3: Model Training (XGBoost, RF, LogReg, NeuralNet)
  Step 3b: Ensemble Building (WeightedVoting selected by log_loss)
  Step 4: Evaluation (accuracy, log_loss, Brier, calibration)
  Step 5: Tournament Simulation (Monte Carlo, 1000 iterations, XGBoost model)
  Step 6: Visualization (power rankings, round probabilities, calibration)
```

```
+------------------+     +-------------------+     +------------------+
|   Data Sources   |---->|  Feature Eng.     |---->|  Model Training  |
| - Kaggle (48k+)  |     | - Elo ratings      |     | - XGBoost        |
| - Wikipedia       |     | - FIFA rankings    |     | - Random Forest  |
| - ESPN            |     | - Form (5/10/EWM)  |     | - Logistic Regr. |
| - the-odds-api    |     | - SOS, H2H, draw    |     | - NeuralNet (MLP)|
| - Transfermarkt   |     | - Odds (implied)   |     +--------+---------+
+------------------+     | - Squad quality     |              |
                           | - 80 features total |              |
                           +---------+----------+              |
                                     |                         v
                                     |              +-----------+-----------+
                            |              |  Ensemble (selected by log_loss) |
                            |              |  from Voting/WeightedVoting/     |
                            |              |  Stacking candidates             |
                                     |              +-----------+-----------+
                                     |                          |
                           +---------+----------+               |
                           |  WC 2026 Features   |<-------------+
                           |  (per-match vectors) |
                           +---------+------------+
                                     |
                                     v
                           +---------------------+
                           |  Monte Carlo Sim    |
                           |  (1,000 runs)       |
                           |  - XGBoost model    |
                           |  - Imputer loaded   |
                           |  - Group stage      |
                           |  - Knockout bracket |
                           |  - Best 3rd-place   |
                           +---------+-----------+
                                     |
                                     v
                           +---------------------+
                           |  Visualization       |
                           |  - Power rankings   |
                           |  - Group heatmaps   |
                           |  - Round probs      |
                           +---------------------+
```

## Features

- **Multi-source data collection**: Kaggle match results, Wikipedia groups/fixtures/rankings, ESPN match odds, the-odds-api outright odds, Transfermarkt squad quality data
- **Elo rating system**: Tournament-specific K-factors (WC=80, Qualifier=60, Friendly=40), home advantage adjustment, draw probability modeling with `ELO_DRAW_FACTOR=0.30`
- **80 features per match**: Elo (7), FIFA rankings (7), form (28: last 5/10 win/draw/loss rates, goals, EWM), H2H (6), SOS (2), draw-predictive (5), context (7), interaction (2), odds (3), squad quality (8)
- **4 ML models**: XGBoost (Optuna-tuned for log_loss), Random Forest (GridSearch, `class_weight="balanced"`, `max_depth=20`), Logistic Regression (balanced, C-tuned), NeuralNet (sklearn MLPClassifier, 8x draw weight)
- **Ensemble selection**: 4 base models + ensemble candidates (VotingEnsemble, WeightedVotingEnsemble with inverse-log-loss weights, StackingClassifier with LogReg meta-learner), best selected by validation log_loss; current best is WeightedVotingEnsemble
- **Draw class weighting**: XGBoost uses 4x sample weight for draws (default `_compute_sample_weights`), NeuralNet uses 8x; RF and LogReg use `class_weight="balanced"`; meta-learner does NOT use class_weight
- **WC draw calibration**: `WC_GROUP_DRAW_RATE=0.25` boosts under-predicted draws in group stage simulation toward historical ~25% rate
- **Probability bug fixes**: Corrected `predict_proba()` output ordering (`[away_win, draw, home_win]` -> `[home_win, draw, away_win]`), added probability normalization, loaded trained imputer instead of `np.nanmean()`, fixed fixture swap bug (swapped probabilities when match lookup is reversed)
- **Monte Carlo simulation**: 1,000 full tournament runs through group stage and knockout bracket (configurable), using XGBoost model for speed
- **2026 format support**: 48 teams, 12 groups (A-L), top 2 + 8 best 3rd-place advance (32-team knockout)
- **Neutral venue handling**: WC 2026 matches correctly flagged as neutral except for host nation home matches
- **Live validation**: Auto-scrape live WC2026 results and validate predictions; manual override CSV supported

## Evaluation Results (2022 WC, 64 matches)

| Metric   | Before  | After   |
|----------|---------|---------|
| Accuracy | 56.25%  | 59.38%  |
| Log Loss | 0.9789  | 0.9383  |
| Avg Brier| 0.1913  | 0.1657  |

These improvements come from draw class weighting (4x for XGBoost, 8x for NeuralNet, `class_weight="balanced"` for RF/LogReg, no class_weight on meta-learner), `ELO_DRAW_FACTOR=0.30`, draw-predictive features (`elo_close`, `draw_tendency`, `fifa_close`), squad quality features, ensemble/model selection by log_loss, and fixture swap bug fix.

## Model Performance (test set)

| Model | Accuracy | Log Loss |
|---|---|---|
| BestEnsemble (WeightedVoting) | 0.615 | 0.835 |
| RandomForest | 0.593 | 0.857 |
| LogisticRegression | 0.579 | 0.874 |
| XGBoost | 0.505 | 0.955 |
| NeuralNet | 0.359 | 1.732 |

### Live Validation (WC 2026, 13 matches)

| Model | Accuracy | Log Loss |
|---|---|---|
| BestEnsemble | 46.2% (6/13) | 1.07 |

Note: LightGBM was removed from the ensemble due to poor performance (0.49 accuracy, 0.99 log_loss). Calibration via `CalibratedWrapper` was tested but removed because it hurt test performance (log_loss 0.8374 -> 1.0436). The model is well-calibrated but argmax rarely picks draw; all 5 actual draws were missed (Canada-BIH, Qatar-SUI, Brazil-MAR, Netherlands-JPN, Spain-CPV).

## Simulation Results (WC 2026, 1000 runs with XGBoost)

Top 5: Mexico 7.9%, Switzerland 5.6%, USA 5.1%, Ivory Coast 4.8%, France 4.6%. Host nations (Mexico, USA, Canada) collectively 17.3%.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)**: Fast Python package manager
- **Kaggle API**: Credentials at `~/.kaggle/kaggle.json` (set up interactively on first run)
- **the-odds-api key** (optional): Set `ODDS_API_KEY` environment variable for outright odds; free at https://the-odds-api.com/

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd worldcup

# Install dependencies using uv
uv sync

# Or install with dev dependencies
uv sync --group dev
```

The `uv sync` command reads `pyproject.toml` and `uv.lock`, creates a virtual environment, and installs all dependencies.

## Quick Start

```bash
# Set up environment (install deps, create directories)
python run_pipeline.py --setup-only

# Run the full pipeline
python run_pipeline.py --all

# Run with custom simulation count
python run_pipeline.py --all --n-simulations 5000

# Run with live validation during the tournament
python run_pipeline.py --all --live-validate

# Retrain models incorporating live WC2026 results
python run_pipeline.py --all --retrain
```

The first run will prompt for Kaggle API credentials if `~/.kaggle/kaggle.json` is not found.

## CLI Usage

```
python run_pipeline.py [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--all` | Run the complete pipeline (scraping through visualization) |
| `--step STEP` | Run a specific step: `scraping`, `features`, `train`, `ensemble`, `evaluate`, `simulate`, `visualize`, `live-validate` |
| `--retrain` | Include live WC2026 data when training models |
| `--live-validate` | Validate model predictions against played WC2026 matches |
| `--n-simulations N` | Override the number of Monte Carlo simulations (default: 1000) |
| `--setup-only` | Only set up the environment (create directories, install deps) |
| `--debug` | Enable debug logging |

### Examples

```bash
# Run just the scraping step
python run_pipeline.py --step scraping

# Retrain models with live data
python run_pipeline.py --step train --retrain

# Run 50,000 simulations
python run_pipeline.py --step simulate --n-simulations 50000

# Validate against live results
python run_pipeline.py --step live-validate
```

## Project Structure

```
worldcup/
├── pyproject.toml                   # uv project config, dependencies, ruff/pytest settings
├── run_pipeline.py                   # Main CLI orchestrator
├── data/
│   ├── external/
│   │   └── continents.csv           # 211 countries to confederation mapping
│   ├── raw/                          # Scraped/downloaded data (gitignored)
│   │   ├── international_matches/    # Kaggle match results + shootouts
│   │   ├── fifa_rankings/           # Historical FIFA rankings from Kaggle
│   │   ├── fifa_rankings_current.csv # Scraped current rankings
│   │   ├── fifa_rankings_merged.csv  # Merged historical + current rankings
│   │   ├── wc2026_groups.csv        # 48 teams in 12 groups
│   │   ├── wc2026_fixtures.csv      # Scheduled match fixtures (72 matches)
│   │   ├── odds_outright.csv        # Tournament winner odds
│   │   ├── odds_match.csv           # Per-match betting odds
│   │   ├── wc2026_results_live.csv  # Live/auto-scraped results
│   │   ├── wc2026_results_manual.csv # Manual results override template
│   │   ├── historical_world_cups.csv # Historical WC bracket data
│   │   └── squad_quality.csv        # Transfermarkt squad quality data (48 teams)
│   └── processed/                    # Engineered features, models, results
│       ├── match_features.parquet    # 48k+ match feature vectors (80 features)
│       ├── wc2026_match_features.parquet  # WC2026 match features
│       ├── elo_ratings_current.parquet    # Current Elo ratings
│       ├── elo_ratings_history.parquet    # Elo rating history
│       ├── models/                   # Trained model files (.joblib)
│       │   ├── xgboost.joblib
│       │   ├── random_forest.joblib
│       │   ├── logistic_regression.joblib
│       │   ├── neural_net.joblib      # sklearn MLPClassifier
│       │   ├── best_model.joblib     # Best ensemble (WeightedVotingEnsemble by log_loss, 96MB)
│       │   ├── imputer.joblib        # Fitted SimpleImputer (median strategy)
│       │   └── feature_columns.joblib
│       ├── evaluation/               # Model evaluation outputs
│       │   ├── model_comparison.csv
│       │   ├── evaluation_report.json
│       │   ├── calibration_curves.png
│       │   ├── feature_importance.png
│       │   ├── tournament_probabilities.png
│       │   └── round_probabilities.png
│       ├── tournament_probabilities.csv
│       ├── group_stage_probabilities.csv
│       └── live_validation_report.csv
├── src/
│   ├── __init__.py
│   ├── config.py                    # Central configuration constants
│   ├── helpers.py                   # Logging, Kaggle setup, team normalization
│   ├── scraping/
│   │   ├── __init__.py
│   │   ├── download_kaggle.py       # Kaggle API dataset downloads
│   │   ├── scrape_fifa_rankings.py  # Current FIFA rankings from Wikipedia
│   │   ├── scrape_world_cup_2026.py # WC2026 groups + fixtures from Wikipedia
│   │   ├── scrape_odds.py           # Bookmaker odds (the-odds-api + ESPN)
│   │   ├── scrape_live_results.py   # Live WC2026 match results
│   │   ├── scrape_historical_world_cups.py # Historical WC brackets
│   │   └── scrape_squad_quality.py  # Transfermarkt squad quality data
│   ├── features/
│   │   ├── __init__.py
│   │   ├── elo.py                   # Elo rating system (draw factor 0.30)
│   │   ├── build_features.py        # Match-level feature engineering (cache v7)
│   │   └── build_2026_features.py   # WC2026 match feature vectors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                 # XGBoost, RF, LogReg, NeuralNet training
│   │   ├── ensemble.py              # Ensemble selection (Voting, WeightedVoting, Stacking) by log_loss
│   │   ├── evaluate.py              # Model evaluation and comparison
│   │   ├── live_validation.py       # Validate against played WC2026 matches
│   │   └── prediction.py            # predict_with_draw_threshold(), predict_proba_with_draw_boost()
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── group_stage.py           # Group stage Monte Carlo simulation (imputer + probability fix + draw calibration + swap fix)
│   │   ├── knockout_stage.py       # Knockout bracket simulation (imputer + probability fix + swap fix)
│   │   └── simulator.py             # Full tournament orchestrator (XGBoost default + draw calibration + swap fix)
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py                 # Matplotlib/seaborn visualizations
│       └── tables.py                # Formatted text tables
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   ├── 03_tournament_simulation.ipynb
│   ├── 04_power_rankings.ipynb
│   └── 05_live_validation.ipynb
└── tests/
    ├── conftest.py                   # Shared pytest fixtures
    ├── test_elo.py                   # Elo rating system tests
    ├── test_features.py              # Feature engineering tests
    ├── test_models.py                # Model training and ensemble tests
    └── test_simulation.py            # Simulation and tournament tests
```

## Data Sources

| Source | Data | Format | Access | Rows/Entries |
|--------|------|--------|--------|--------------|
| Kaggle (`martj42/international-football-results-from-1872-to-2017`) | International match results (1872-present) | CSV | Kaggle API | 48,000+ |
| Kaggle (`cashncarry/fifaworldranking`) | Historical FIFA rankings (1993-present) | CSV | Kaggle API | 60,000+ |
| Wikipedia (`2026_FIFA_World_Cup`) | Group compositions, fixtures, current rankings | HTML/wikitext | HTTP scrape | 48 teams |
| ESPN | Per-match betting odds | HTML | HTTP scrape | Variable |
| the-odds-api | Outright tournament winner odds | JSON API | API key | Variable |
| Wikipedia (historical WC pages) | Historical World Cup brackets | Wikitext | HTTP scrape | 1930-2022 |
| Transfermarkt | Squad quality data (market value, size, age) for 48 WC 2026 teams | HTML | HTTP scrape | 48 teams |

## Model Architecture

### Base Models (4 models; LightGBM removed)

| Model | Tuning | Key Parameters |
|-------|--------|----------------|
| **XGBoost** | Optuna (20 trials, 3-fold TimeSeriesSplit CV, optimized for log_loss) | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda; draw class weight 4x |
| **Random Forest** | GridSearch (3-fold TimeSeriesSplit, scoring=neg_log_loss, class_weight="balanced") | n_estimators=[100,200], max_depth=[10,20], min_samples_split=[2,5] |
| **Logistic Regression** | GridSearchCV (3-fold TimeSeriesSplit, scoring=neg_log_loss, class_weight="balanced") | C=[0.01,0.1,1.0,10.0], StandardScaler pipeline |
| **NeuralNet (sklearn MLP)** | Early stopping (patience=10) | Layers [128, 64, 32], alpha=0.001, batch_size=256, adaptive lr, max_iter=300; draw class weight 8x |

### Ensemble Strategy

The ensemble builder (`build_best_ensemble`) evaluates multiple candidates and selects the best by validation **log_loss** (not accuracy):

1. **Individual models**: Each base model evaluated on validation set
2. **VotingEnsemble (uniform)**: Soft voting with equal weights
3. **WeightedVotingEnsemble**: Soft voting with inverse-log-loss weights
4. **StackingEnsemble**: LogisticRegression meta-learner (no class_weight) over all base models

The candidate with the lowest validation log_loss is saved as `best_model.joblib`. The current best ensemble is a WeightedVotingEnsemble (soft voting with inverse-log-loss weights across 4 base models).

### Why LightGBM Was Removed

LightGBM had the worst performance of all models (accuracy 0.49, log_loss 0.99) and was removed from the pipeline. The `train_lightgbm()` function is preserved in `train.py` for reference but is not called during training.

### Why Calibration Was Removed

Isotonic calibration via `CalibratedWrapper` was tested but degraded test performance (log_loss 0.8374 -> 1.0436). The `CalibratedWrapper` class is preserved in `ensemble.py` for future experimentation.

### Draw Handling

Draws are the hardest outcome to predict in football. The pipeline uses multiple strategies:

- **ELO_DRAW_FACTOR = 0.30**: Scales draw probability in the Elo system (increased from 0.25 for better calibration)
- **WC_GROUP_DRAW_RATE = 0.25**: Historical WC group-stage draw rate, used for draw calibration in simulation
- **XGBoost sample weights**: `away_win=1.0, draw=4.0, home_win=1.0` (4x weight on draws, default from `_compute_sample_weights`)
- **NeuralNet sample weights**: `away_win=1.0, draw=8.0, home_win=1.0` (8x weight on draws)
- **RF and LogReg**: `class_weight="balanced"` for automatic class rebalancing
- **Stacking meta-learner**: LogisticRegression without class_weight (removed — was hurting log_loss)
- **Draw-predictive features**: `elo_close` (Elo delta < 100), `draw_tendency` (combined draw prob amplified for close matches), `fifa_close` (FIFA rank delta < 20), `tournament_draw_rate`, `combined_draw_prob`
- **Ensemble selection by log_loss**: Avoids penalizing draw predictions that improve probability calibration

### Label Encoding

| Outcome | Original | Mapped |
|---------|----------|--------|
| Home win | 1 | 2 |
| Draw | 0 | 1 |
| Away win | -1 | 0 |

Important: `predict_proba()` returns probabilities in class order `[away_win(0), draw(1), home_win(2)]`, which must be reordered to `[home_win, draw, away_win]` for simulation.

## Feature Engineering (80 features)

### Elo Features (7)

| Feature | Description |
|---------|-------------|
| `elo_home` | Home team Elo rating |
| `elo_away` | Away team Elo rating |
| `elo_delta` | Elo difference (home - away) |
| `elo_abs_delta` | Absolute Elo difference |
| `elo_home_win_prob` | Elo-predicted home win probability |
| `elo_draw_prob` | Elo-predicted draw probability |
| `elo_away_win_prob` | Elo-predicted away win probability |

### FIFA Rankings (7)

| Feature | Description |
|---------|-------------|
| `fifa_rank_home` | Home team FIFA ranking (lower is better) |
| `fifa_rank_away` | Away team FIFA ranking |
| `fifa_rank_delta` | Ranking difference (home - away) |
| `fifa_rank_abs_delta` | Absolute ranking difference |
| `fifa_points_home` | Home team FIFA ranking points |
| `fifa_points_away` | Away team FIFA ranking points |
| `fifa_points_delta` | Points difference |
| `fifa_points_abs_delta` | Absolute points difference |

### Form Features (28: last 5/10 standard + EWM)

| Feature | Description |
|---------|-------------|
| `home_form_last10_*` | Home team last-10-match form (7 metrics: win/draw/loss rate, goals scored/conceded avg, goal diff avg, clean sheet rate) |
| `home_form_last5_*` | Home team last-5-match form (same 7 metrics) |
| `away_form_last10_*` | Away team last-10-match form |
| `away_form_last5_*` | Away team last-5-match form |
| `away_form_last10_ewm_*` | Away team last-10 EWM form (3 metrics: win/draw/loss rate) |
| `away_form_last5_ewm_*` | Away team last-5 EWM form (3 metrics) |

### Head-to-Head (6)

| Feature | Description |
|---------|-------------|
| `h2h_home_wins` | Head-to-head home wins (last 5 meetings) |
| `h2h_draws` | Head-to-head draws |
| `h2h_away_wins` | Head-to-head away wins |
| `h2h_draw_rate` | Head-to-head draw rate |
| `h2h_home_goals_avg` | Head-to-head average home goals |
| `h2h_away_goals_avg` | Head-to-head average away goals |

### Strength of Schedule (2)

| Feature | Description |
|---------|-------------|
| `home_sos_avg_opp_elo` | Home team strength of schedule (average opponent Elo) |
| `away_sos_avg_opp_elo` | Away team strength of schedule |

### Draw-Predictive Features (5)

| Feature | Description |
|---------|-------------|
| `elo_close` | Binary: 1 if abs(Elo delta) < 100 |
| `draw_tendency` | Combined draw prob amplified for close matches |
| `fifa_close` | Binary: 1 if abs(FIFA rank delta) < 20 |
| `tournament_draw_rate` | Historical draw rate for the tournament type |
| `combined_draw_prob` | Blended draw probability from Elo and form |

### Context Features (7)

| Feature | Description |
|---------|-------------|
| `neutral` | Whether match is on neutral ground (0/1) |
| `home_advantage` | Home advantage weight (1.0 home, 0.5 neutral) |
| `is_host_nation` | Whether home team is a host nation (US/CA/MX) |
| `same_confederation` | Whether both teams share a confederation |
| `is_world_cup` | Whether match is a World Cup match |
| `is_qualifier` | Whether match is a World Cup qualifier |
| `is_knockout` | Whether match is a World Cup knockout match (date-based detection) |

### Interaction Features (2)

| Feature | Description |
|---------|-------------|
| `elo_delta_x_home_advantage` | Interaction: Elo delta multiplied by home advantage |
| `fifa_rank_delta_x_same_confed` | Interaction: Rank delta multiplied by same-confederation flag |

### Odds Features (3, when available)

| Feature | Description |
|---------|-------------|
| `odds_home_implied_prob` | Implied probability from bookmaker home odds (WC2026 only) |
| `odds_draw_implied_prob` | Implied probability from bookmaker draw odds |
| `odds_away_implied_prob` | Implied probability from bookmaker away odds |

### Squad Quality Features (8)

| Feature | Description |
|---------|-------------|
| `home_squad_value_m` | Home team total squad market value (€M, from Transfermarkt) |
| `away_squad_value_m` | Away team total squad market value (€M) |
| `squad_value_delta` | Squad value difference: home - away (€M) |
| `squad_value_abs_delta` | Absolute squad value difference (€M) |
| `home_avg_player_value_m` | Home team average player market value (€M) |
| `away_avg_player_value_m` | Away team average player market value (€M) |
| `home_top_player_value_m` | Home team most valuable player (€M) |
| `away_top_player_value_m` | Away team most valuable player (€M) |

Historical matches use confederation-average fallbacks for missing squad quality data (66% have actual data, 34% use confed avg). All WC 2026 matches have actual Transfermarkt data.

## Simulation Methodology

The tournament simulation uses Monte Carlo methods to estimate advancement and winning probabilities. XGBoost is used as the default simulation model for speed (819KB compressed, fast inference).

### Probability Handling (Bug Fixes)

Two critical bugs were fixed in the simulation code:

1. **Probability mapping**: `predict_proba()` returns `[away_win, draw, home_win]` (classes 0, 1, 2), but the simulator previously treated it as `[home_win, draw, away_win]`. Fixed to reorder: `return np.array([proba[2], proba[1], proba[0]])`.
2. **Missing imputer**: Both simulators used `np.nanmean()` to fill NaN features instead of the trained `SimpleImputer` (per-feature medians). Now loads and uses `imputer.joblib`.
3. **Probability normalization**: Added `probs = probs / probs.sum()` before sampling to prevent `ValueError: Probabilities do not sum to 1`.
4. **Knockout draw redistribution**: Fixed index mapping after probability reorder -- `probs[0]` is now home_win, `probs[1]` is draw, `probs[2]` is away_win.
5. **Fixture swap bug (CRITICAL)**: When a fixture lookup falls back to the reversed match, probabilities were from the wrong perspective. Fixed by adding a `swapped` flag and swapping `[home_win, draw, away_win]` → `[away_win, draw, home_win]` in 4 files: `group_stage.py`, `knockout_stage.py`, `simulator.py`, `live_validation.py`.

### Group Stage

1. For each of the 12 groups (A-L, 4 teams each), simulate all 6 round-robin matches
2. For each match, the model predicts win/draw/loss probabilities; NaN features are imputed with the trained SimpleImputer; probabilities are normalized and reordered; fixture swaps are handled
3. **Draw calibration** (group stage only): If the model's predicted draw probability is below `WC_GROUP_DRAW_RATE=0.25`, it is boosted to 25% with excess redistributed proportionally from win/loss probabilities
4. Goal totals are generated using Poisson distributions (mean 1.5 for winners, 1.0 for draws)
4. Teams are ranked by points, then goal difference, then goals scored
5. Top 2 from each group advance; best 8 third-place teams also advance

### Knockout Stage

1. Draw probability is redistributed to win/loss (70% reduction, 55/45 split favoring the stronger team based on model probabilities). Draw calibration is NOT applied here.
2. Each knockout match produces a single winner (no draws)
3. Bracket proceeds through Round of 32, Round of 16, Quarter-finals, Semi-finals, Final

### Aggregation

- Run 1,000 complete tournaments by default (configurable via `--n-simulations`)
- Track how far each team advances in each run
- Compute probabilities for each round advancement and tournament winner

### Live Results

- When live WC2026 results are available, group stage standings are updated with actual scores
- Remaining matches in the group are still simulated
- This grounds the simulation in reality as the tournament progresses

## Bug Fixes (v0.3.0)

### Critical Bug Fixes

| Bug | Impact | Fix |
|-----|--------|-----|
| Probability mapping in simulation | Model `predict_proba()` returns `[away_win, draw, home_win]` but simulator treated it as `[home_win, draw, away_win]`, causing inverted home/away predictions | Reordered output: `np.array([proba[2], proba[1], proba[0]])` in both `group_stage.py` and `knockout_stage.py` |
| Missing imputer in simulation | Both simulators used `np.nanmean()` (global mean) instead of trained `SimpleImputer` (per-feature medians) | Load `imputer.joblib` and use `imputer.transform()` in both simulators |
| Probability normalization | Probabilities could drift from summing to 1.0, causing `ValueError: Probabilities do not sum to 1` | Added `probs = probs / probs.sum()` before sampling in both simulators |
| Knockout draw redistribution | After probability reorder, draw redistribution used wrong indices (`probs[1]` was draw, not `probs[0]`) | Fixed: `probs[0]` = home_win, `probs[1]` = draw, `probs[2]` = away_win |
| `features_list.append(features)` missing | Empty DataFrames, `KeyError: 'elo_delta'` | Features now correctly appended in both `build_features.py` and `build_2026_features.py` |
| `neutral=0` for all WC 2026 matches | Home advantage wrongly applied to neutral venues | Neutral venues get `neutral=1`; host nation matches get `neutral=0` with `home_advantage=1.0` |
| Stale Elo cache | Only 4 teams in cache, predictions used default ratings | Cache recomputed from full match history if fewer than 10 teams |
| WC fixture duplicates | 144 matches (permutations) instead of 72 (combinations) | Fixed to use `combinations` for correct 6 matches per group x 12 groups |
| `is_knockout` barely triggered | Only 3/49413 matches flagged as knockout | Added date-based WC knockout detection; 271 matches now correctly flagged |
| **Fixture swap bug** | Probabilities from wrong perspective when match lookup is reversed | Added `swapped` flag + probability swap in 4 files: `group_stage.py`, `knockout_stage.py`, `simulator.py`, `live_validation.py` |

### Model Changes

| Change | Before | After |
|--------|--------|-------|
| LightGBM | Included (0.49 accuracy, 0.99 log_loss) | Removed (worst performer) |
| Best ensemble | Voting/Stacking selection | WeightedVotingEnsemble (inverse-log-loss weighted soft voting, selected by log_loss) |
| Calibration | Tested CalibratedWrapper (isotonic) | Removed (hurt test log_loss: 0.8374 -> 1.0436) |
| Feature cache version | "5" | "7" (added squad quality features) |
| Simulation model | Best ensemble | XGBoost (fast, compressed with compress=3) |
| Draw class weight (XGBoost) | 1.5x | 4.0x (default `_compute_sample_weights`) |
| Draw class weight (NeuralNet) | 4.0x | 8.0x (overrides default) |
| Stacking meta-learner class_weight | None | Removed (was "balanced", removed for better log_loss) |
| ELO_DRAW_FACTOR | 0.25 | 0.30 |
| WC draw calibration | Not present | WC_GROUP_DRAW_RATE=0.25, boosts under-predicted draws in group stage |
| Fixture swap bug | Probabilities from wrong perspective | `swapped` flag + probability swap in 4 files |
| Validation split | val_years=[2022] | val_years=[2023] |
| RF max_depth | None (unlimited) | 20 (capped) |
| RF n_estimators grid | [100,200,300] | [100,200] |
| OPTUNA_TRIALS | 100 | 20 |
| CV_FOLDS | 5 | 3 |
| Model compression | No compression | compress=3 on all joblib.dump calls |
| PyTorch import | import torch.nn as nn (~1GB) | Removed (was causing kernel death/OOM) |

### Model Optimizations

| Change | Before | After |
|--------|--------|-------|
| Draw sample weight (XGBoost) | 1.5x | 4.0x |
| Draw sample weight (NeuralNet) | 4.0x | 8.0x |
| RF class_weight | None | `"balanced"` |
| LogReg class_weight | None | `"balanced"` + C tuning |
| Stacking meta-learner class_weight | None | Removed (was set to "balanced", now no class_weight) |
| ELO_DRAW_FACTOR | 0.25 | 0.30 |
| WC draw calibration | Not present | `WC_GROUP_DRAW_RATE=0.25` in group stage |
| Ensemble selection metric | accuracy | log_loss |
| XGBoost Optuna scoring | accuracy | log_loss |
| RF GridSearch scoring | accuracy | neg_log_loss |
| Neural Net alpha | 0.0001 | 0.001 |
| Neural Net batch_size | 64 | 256 |
| Neural Net learning rate | fixed | adaptive |
| Neural Net max_iter | 200 | 300 |
| RF max_depth | None (unlimited) | 20 (capped) |
| RF n_estimators grid | [100,200,300] | [100,200] |
| OPTUNA_TRIALS | 100 | 20 |
| CV_FOLDS | 5 | 3 |
| Feature cache version | "1" | "7" |
| Model compression | No compression | compress=3 on all joblib.dump calls |
| Validation split | val_years=[2022] | val_years=[2023] |
| PyTorch import | import torch.nn as nn (~1GB) | Removed (kernel death/OOM) |

## Live Validation

During the tournament, the pipeline can validate predictions against actual results:

```bash
python run_pipeline.py --step live-validate
```

- Auto-scrapes live match results from ESPN/Wikipedia
- Compares model predictions to actual outcomes
- Reports accuracy (argmax and draw-threshold), log-loss on played matches
- Supports manual overrides via `data/raw/wc2026_results_manual.csv`
- Returns `accuracy_argmax`, `accuracy_threshold`, `correct_argmax`, `correct_threshold`, `draw_ratio` per match

### Manual Override CSV Format

```csv
date,home_team,away_team,home_score,away_score,group,match_number
2026-06-11,Mexico,New Zealand,3,0,A,1
```

## Configuration

All configuration constants are in `src/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `RANDOM_STATE` | 42 | Random seed for reproducibility |
| `N_SIMULATIONS` | 1000 | Number of Monte Carlo tournament simulations |
| `ELO_HOME_ADVANTAGE` | 100 | Elo rating bonus for home team |
| `ELO_INITIAL_RATING` | 1000 | Starting Elo for new teams |
| `ELO_DRAW_FACTOR` | 0.30 | Draw probability scaling factor |
| `WC_GROUP_DRAW_RATE` | 0.25 | Historical WC group-stage draw rate for draw calibration |
| `K_FACTORS` | WC: 80, Qual: 60, Friendly: 40, default: 50 | Elo K-factor by tournament type |
| `NEURAL_NET_EPOCHS` | 100 | Max training iterations for sklearn MLP (overridden to 300 in build) |
| `NEURAL_NET_PATIENCE` | 10 | Early stopping patience for MLP |
| `NEURAL_NET_LAYERS` | [128, 64, 32] | MLP hidden layer sizes |
| `NEURAL_NET_DROPOUT` | 0.3 | Not used (sklearn MLP uses alpha for regularization); import removed from train.py |
| `NEURAL_NET_LEARNING_RATE` | 1e-3 | MLP initial learning rate |
| `OPTUNA_TRIALS` | 20 | Number of Optuna hyperparameter trials |
| `CV_FOLDS` | 3 | Cross-validation folds |
| `HOST_NATIONS` | [US, CA, MX] | 2026 host countries |
| `N_GROUPS` | 12 | Number of groups (A-L) |
| `ADVANCE_PER_GROUP` | 2 | Teams advancing per group |
| `BEST_THIRD_ADVANCE` | 8 | Best third-place teams advancing |

### Draw Class Weights

Used in `_compute_sample_weights()` in `train.py`:

| Class | Label | Weight |
|-------|-------|--------|
| Away win | 0 | 1.0 |
| Draw | 1 | 4.0 (XGBoost) / 8.0 (NeuralNet) |
| Home win | 2 | 1.0 |

Note: `_compute_sample_weights()` defaults to 4x for draw class. XGBoost uses the default (4x), while NeuralNet overrides to 8x. Random Forest and Logistic Regression use `class_weight="balanced"` instead of explicit sample weights. The Stacking ensemble meta-learner does NOT use class_weight (removed for better log_loss).

## Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_elo.py

# Run with verbose output
uv run pytest -v

# Run a specific test class
uv run pytest tests/test_models.py::TestModelTraining -v
```

Tests cover (35 tests total):
- **test_elo.py** (8 tests): Elo initialization, computation, probability bounds, K-factors, rating retrieval
- **test_features.py** (5 tests): Feature column presence, outcome encoding, value ranges, name normalization
- **test_models.py** (6 tests): Prediction shapes, probability sums, label mapping, ensemble behavior
- **test_simulation.py** (7 tests): Group structure, knockout match resolution, probability monotonicity, bounds
- **Other tests** (9 tests): Integration, config, helpers, scraping, etc.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Load and explore raw data; visualize distributions, correlations, missing values |
| `02_model_analysis.ipynb` | Compare model performance; feature importance analysis; calibration curves (loads/evaluates models one at a time with `del`+`gc.collect()` to avoid OOM) |
| `03_tournament_simulation.ipynb` | Run simulations interactively; explore bracket paths; analyze group outcomes |
| `04_power_rankings.ipynb` | Generate and visualize power rankings; group-stage heatmaps |
| `05_live_validation.ipynb` | Compare predictions vs live results; track accuracy over time (uses new `accuracy_argmax`/`accuracy_threshold` keys) |

All notebooks use the `worldcup-2026-predictor` kernel (registered from the `uv` venv) to avoid NumPy version incompatibility with saved models.

## License

All rights reserved. License to be determined.