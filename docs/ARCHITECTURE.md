# Architecture

This document describes the system architecture, data flow, module structure, and algorithms used in the World Cup 2026 ML Predictor.

## System Architecture Overview

The pipeline follows a six-stage sequential process, with each stage producing artifacts consumed by subsequent stages. All data artifacts are persisted to `data/raw/` or `data/processed/` as Parquet/CSV/Joblib files, enabling resumable execution via the `--step` CLI flag.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_pipeline.py (CLI)                        │
│  --all | --step {scraping,features,train,ensemble,evaluate,       │
│                   simulate,visualize,live-validate}                 │
│  --retrain | --live-validate | --n-simulations N | --setup-only    │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────┘
       │          │          │          │          │          │
       v          v          v          v          v          v
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │Step 1  │ │Step 2  │ │Step 3  │ │Step 4  │ │Step 5  │ │Step 6  │
  │Scraping│ │Features│ │Training│ │Evaluate│ │Simulate│ │ Visual │
  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
      │          │          │          │          │          │
      v          v          v          v          v          v
  data/raw/  data/     data/      data/      data/      data/
             processed/ processed/ processed/ processed/ processed/
             match_     models/    evaluation/ tournament  evaluation/
             features   *.joblib   *.csv      _probs.csv  *.png
             .parquet   imputer               group_
                                               stage_
                                               probs.csv
```

## Data Flow

```
   Kaggle API ──┐
   Wikipedia ───┤     ┌──────────────────┐     ┌──────────────────┐
   ESPN ────────┤────>│  data/raw/       │────>│  Feature Eng.      │
   the-odds-api─┤     │  *.csv, *.json   │     │  src/features/     │
    Transfermarkt┤     └──────────────────┘     └────────┬──────────┘
    (squad quality)│                                                       
   Manual CSV ──┘                                       │
                                                           │
                                                           v
                          ┌──────────────────┐     ┌──────────────────┐
                          │  Match Features    │<────│  Elo Ratings      │
                          │  .parquet (48k+)   │     │  .parquet          │
                          └────────┬──────────┘     └──────────────────┘
                                   │
                     ┌─────────────┼─────────────┐
                     v             v              v
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ XGBoost  │  │ RandForest│  │ LogReg   │  ┌──────────┐
              │ (Optuna)  │  │ (GridSea.)│  │ (balanced)│  │ MLP      │
              └─────┬────┘  └─────┬─────┘  └─────┬────┘  │ (sklearn)│
                    │             │              │       └─────┬────┘
                    └─────────┬──┴──────────────┴─────────────┘
                              v
                     ┌──────────────────┐
                     │  StackingEnsemble  │
                     │  (4 base models +  │
                     │  LogReg meta)       │
                     │  Selected by        │
                     │  validation log_loss │
                     └────────┬──────────┘
                              │
                              v
                     ┌──────────────────┐     ┌──────────────────┐
                     │  best_model.joblib│────>│  Monte Carlo Sim  │
                     │  feature_columns  │     │  (XGBoost, 1k)    │
                     │  imputer.joblib   │     └────────┬──────────┘
                     └──────────────────┘              │
                          ┌───────────────────────────┼──────────────┐
                          v                           v              v
                  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
                  │  Group Stage  │  │  Knockout Stage   │  │  Live Valid. │
                  │  Probs (CSV)  │  │  Bracket (CSV)   │  │  Report (CSV)│
                  └──────────────┘  └──────────────────┘  └──────────────┘
                                                          │
                                                          v
                                                 ┌──────────────────┐
                                                 │  Visualization    │
                                                 │  PNG, text tables│
                                                 └──────────────────┘
```

## Module Descriptions

### `src/scraping/`

Data acquisition from external sources. Each module is idempotent (skips if data already exists) and saves to `data/raw/`.

| Module | Purpose | Output Files |
|--------|---------|--------------|
| `download_kaggle.py` | Download international match results and FIFA rankings via Kaggle API | `data/raw/international_matches/results.csv`, `data/raw/fifa_rankings/fifa_ranking.csv` |
| `scrape_fifa_rankings.py` | Scrape current FIFA rankings from Wikipedia; merge with historical | `data/raw/fifa_rankings_current.csv`, `data/raw/fifa_rankings_merged.csv` |
| `scrape_world_cup_2026.py` | Scrape WC2026 group compositions and match fixtures from Wikipedia | `data/raw/wc2026_groups.csv`, `data/raw/wc2026_fixtures.csv` |
| `scrape_odds.py` | Fetch outright odds (the-odds-api) and per-match odds (ESPN) | `data/raw/odds_outright.csv`, `data/raw/odds_match.csv` |
| `scrape_live_results.py` | Scrape live WC2026 results from ESPN/Wikipedia; merge with manual overrides | `data/raw/wc2026_results_live.csv` |
| `scrape_historical_world_cups.py` | Scrape historical WC brackets (1930-2022) from Wikipedia; fallback to Kaggle filter | `data/raw/historical_world_cups.csv` |
| `scrape_squad_quality.py` | Scrape Transfermarkt for 48 WC 2026 teams: squad market value, size, avg age, foreigners, avg/top player value. Uses German-slug name mapping (`TEAM_SLUG_OVERRIDES`) and search name overrides (`TEAM_SEARCH_OVERRIDES`). Rate-limited at 2s between requests. | `data/raw/squad_quality.csv` |

### `src/features/`

Feature engineering for match-level prediction.

| Module | Purpose | Output Files |
|--------|---------|--------------|
| `elo.py` | `EloRatingSystem` class: compute Elo ratings from 48k+ matches; predict match probabilities with `ELO_DRAW_FACTOR=0.30` | `data/processed/elo_ratings_current.parquet`, `data/processed/elo_ratings_history.parquet` |
| `build_features.py` | Build feature vectors for every historical match (80 features each); cache version "7"; includes squad quality features with confederation-average fallbacks | `data/processed/match_features.parquet` |
| `build_2026_features.py` | Build feature vectors for all WC2026 scheduled matches (72 matches); all squad quality data available (0 NaN) | `data/processed/wc2026_match_features.parquet` |

### `src/models/`

Model training, ensemble building, evaluation, and live validation.

| Module | Purpose | Output Files |
|--------|---------|--------------|
| `train.py` | Train XGBoost (Optuna, log loss), RandomForest (GridSearch, balanced), LogisticRegression (balanced, C-tuned), NeuralNet (sklearn MLP); draw class weight 4x; also contains `train_lightgbm()` (not called) | `data/processed/models/*.joblib` |
| `ensemble.py` | Select best ensemble by validation log_loss: evaluates VotingEnsemble, WeightedVotingEnsemble, StackingEnsemble, and individual models. Stacking meta-learner uses `class_weight="balanced"`. Contains `CalibratedWrapper` class (preserved but not used -- calibration hurt test performance) | `data/processed/models/best_model.joblib` |
| `evaluate.py` | Evaluate all models on test set; generate comparison CSV, calibration curves, feature importance | `data/processed/evaluation/` |
| `live_validation.py` | Validate best model predictions against played WC2026 matches | `data/processed/live_validation_report.csv` |

### `src/simulation/`

Monte Carlo tournament simulation. Both simulators load and use the trained `SimpleImputer` (median strategy) for NaN feature handling, correctly reorder `predict_proba()` output from `[away_win, draw, home_win]` to `[home_win, draw, away_win]`, handle swapped fixture lookups by swapping probabilities, and apply draw calibration in group stage but not knockout stage.

| Module | Purpose | Output Files |
|--------|---------|--------------|
| `group_stage.py` | `GroupStageSimulator`: simulate all 12 groups; compute advancement probabilities; determine best 3rd-place qualifiers; normalize probabilities before sampling; includes `_calibrate_draw()` static method to boost draw probability toward historical ~25% WC group-stage rate | (used by simulator) |
| `knockout_stage.py` | `KnockoutStageSimulator`: simulate knockout bracket (R32 through Final); redistribute draw probability with correct index mapping; handles swapped fixture lookups | (used by simulator) |
| `simulator.py` | `WorldCupSimulator`: orchestrate full tournament simulation across N iterations; uses XGBoost by default for speed; incorporate live results; includes `_calibrate_draw()` module-level function and `_predict_match_proba()` with swap handling and draw calibration (group stage only) | `data/processed/tournament_probabilities.csv`, `data/processed/group_stage_probabilities.csv` |

### `src/visualization/`

Plotting and formatted text output.

| Module | Purpose | Output Files |
|--------|---------|--------------|
| `plots.py` | Generate matplotlib/seaborn charts: tournament probabilities, group heatmaps, feature importance, Elo ratings, round advancement, model comparison | `data/processed/evaluation/*.png` |
| `tables.py` | Format text tables: power rankings, group tables, bracket summary | (logged to console) |

## Database / File Storage

The project uses file-based storage exclusively (no SQL database). All intermediate and final data is persisted as:

| Format | Usage | Location |
|--------|-------|----------|
| **Parquet** | Large DataFrames (match features, Elo ratings) | `data/processed/*.parquet` |
| **CSV** | Raw scraped data, results, probabilities | `data/raw/*.csv`, `data/processed/*.csv` |
| **Joblib** | Serialized scikit-learn models, imputer, feature column lists | `data/processed/models/*.joblib` |
| **JSON** | Evaluation reports, API responses | `data/processed/evaluation/*.json`, `data/raw/odds_outright.json` |
| **PNG** | Visualization outputs | `data/processed/evaluation/*.png` |

## Model Training Pipeline

### Data Splitting

The `split_data()` function in `src/models/train.py` uses a time-series split to prevent data leakage:

- **Train**: Matches before 2022
- **Validation**: Matches in 2022 (used for early stopping and ensemble selection)
- **Test**: Matches after 2022 (held out for final evaluation)

This ensures the model never sees future data during training.

### Training Process

```
1. Load match_features.parquet
2. Drop rows with missing outcome
3. Split into train / val / test by year
4. Impute missing values with SimpleImputer (median strategy) -> saved to imputer.joblib
5. Compute sample weights: {away_win: 1.0, draw: 4.0, home_win: 1.0}
6. Train XGBoost with Optuna hyperparameter optimization (log_loss objective)
7. Train Random Forest with GridSearch (neg_log_loss, class_weight="balanced")
8. Train Logistic Regression with C tuning (class_weight="balanced")
9. Train NeuralNet (sklearn MLPClassifier) with sample weights (draw 4x)
10. Save all models + imputer + feature columns to data/processed/models/
```

Note: `train_lightgbm()` is preserved in `train.py` but not called during training, as LightGBM had the worst performance (0.49 accuracy, 0.99 log_loss).

### Feature Selection

The `_get_feature_columns()` function selects features present in the data. There are 27 core features (always present) and 53 optional features (form, EWM form, H2H, SOS, draw-predictive, context, interaction, odds, squad quality). Missing optional features are filled with zeros for WC2026 match predictions.

### Draw Class Weights

XGBoost and NeuralNet use explicit sample weights for the draw class (4x). Random Forest and Logistic Regression use `class_weight="balanced"` for automatic rebalancing. The Stacking ensemble meta-learner also uses `class_weight="balanced"`. This improves draw prediction without sacrificing overall accuracy.

## Ensemble Strategy

### Selection by Log Loss

The `build_best_ensemble()` function in `src/models/ensemble.py` evaluates multiple candidates and selects the one with the lowest validation **log_loss** (not accuracy). This avoids penalizing draw predictions that improve probability calibration.

### Current Best Ensemble

The current best ensemble is a **StackingClassifier** with 4 base models (XGBoost, RandomForest, LogisticRegression, NeuralNet) and a LogisticRegression meta-learner with `class_weight="balanced"`. It achieved the best validation log_loss of all candidates.

### Candidates

1. **Individual models**: XGBoost, RandomForest, LogisticRegression, NeuralNet
2. **VotingEnsemble (uniform)**: Soft voting with equal weights
3. **WeightedVotingEnsemble**: Soft voting with inverse-log-loss weights
4. **StackingEnsemble**: LogisticRegression meta-learner over all base models using `predict_proba`

### Stacking Details

- **Meta-learner**: LogisticRegression (solver=lbfgs, max_iter=1000, class_weight="balanced")
- **Cross-validation**: KFold(n_splits=3, shuffle=True, random_state=42)
- **Stack method**: `predict_proba` (soft probabilities as meta-features)

### Why Log Loss Over Accuracy

Accuracy treats all misclassifications equally, but in football prediction, predicting a draw incorrectly is different from confusing home/away wins. Log loss directly measures the quality of probability predictions, which is critical for:

- Monte Carlo simulation (needs well-calibrated probabilities)
- Brier score optimization
- Distinguishing between close matches and blowouts

### Why LightGBM Was Removed

LightGBM had the worst performance among all models (accuracy 0.49, log_loss 0.99) and was removed from the ensemble. The `train_lightgbm()` function is preserved in the codebase for reference but is not called during training.

### Why Calibration Was Removed

Isotonic calibration via `CalibratedWrapper` was tested to improve probability calibration. However, it degraded test performance significantly (log_loss increased from 0.8374 to 1.0436). The `CalibratedWrapper` class is preserved in `ensemble.py` for future experimentation.

## Feature List

### Core Features (27)

| # | Feature | Type | Source | Description |
|---|---------|------|--------|-------------|
| 1 | `elo_home` | float | Elo system | Home team Elo rating |
| 2 | `elo_away` | float | Elo system | Away team Elo rating |
| 3 | `elo_delta` | float | Derived | Elo difference (home - away) |
| 4 | `elo_abs_delta` | float | Derived | Absolute Elo difference |
| 5 | `elo_home_win_prob` | float | Elo system | Predicted home win probability |
| 6 | `elo_draw_prob` | float | Elo system | Predicted draw probability |
| 7 | `elo_away_win_prob` | float | Elo system | Predicted away win probability |
| 8 | `fifa_rank_home` | int | FIFA rankings | Home team FIFA rank |
| 9 | `fifa_rank_away` | int | FIFA rankings | Away team FIFA rank |
| 10 | `fifa_rank_delta` | int | Derived | Rank difference (home - away) |
| 11 | `fifa_rank_abs_delta` | int | Derived | Absolute rank difference |
| 12 | `fifa_points_home` | float | FIFA rankings | Home team FIFA ranking points |
| 13 | `fifa_points_away` | float | FIFA rankings | Away team FIFA ranking points |
| 14 | `fifa_points_delta` | float | Derived | Points difference |
| 15 | `fifa_points_abs_delta` | float | Derived | Absolute points difference |
| 16 | `neutral` | int | Match data | Neutral venue flag (0/1) |
| 17 | `home_advantage` | float | Derived | 1.0 if home, 0.5 if neutral |
| 18 | `is_host_nation` | int | Config | Home team is host nation (US/CA/MX) |
| 19 | `same_confederation` | int | Confederations | Both teams from same confederation |
| 20 | `is_world_cup` | int | Match data | World Cup match flag |
| 21 | `is_qualifier` | int | Match data | WC qualifier flag |
| 22 | `is_friendly` | int | Match data | Friendly match flag |
| 23 | `is_knockout` | int | Derived | WC knockout match (date-based detection) |
| 24 | `combined_draw_prob` | float | Derived | Blended draw probability from Elo and form |
| 25 | `elo_close` | int | Derived | 1 if abs(Elo delta) < 100 |
| 26 | `draw_tendency` | float | Derived | Combined draw prob amplified for close matches |
| 27 | `fifa_close` | int | Derived | 1 if abs(FIFA rank delta) < 20 |

### Optional Features (53)

| # | Feature | Type | Source | Description |
|---|---------|------|--------|-------------|
| 28-34 | `home_form_last10_*` | float | Computed | Home team last-10 form: win_rate, draw_rate, loss_rate, goals_scored_avg, goals_conceded_avg, goal_diff_avg, clean_sheet_rate |
| 35-41 | `home_form_last5_*` | float | Computed | Home team last-5 form (same metrics) |
| 42-48 | `away_form_last10_*` | float | Computed | Away team last-10 form |
| 49-55 | `away_form_last5_*` | float | Computed | Away team last-5 form |
| 56-61 | `away_form_last10_ewm_*` | float | Computed | Away team last-10 EWM form: ewm_win_rate, ewm_draw_rate, ewm_loss_rate |
| 62-64 | `away_form_last5_ewm_*` | float | Computed | Away team last-5 EWM form: ewm_win_rate, ewm_draw_rate, ewm_loss_rate |
| 65-70 | `h2h_*` | float | Computed | Head-to-head: home_wins, draws, away_wins, draw_rate, home_goals_avg, away_goals_avg (last 5) |
| 71-72 | `home_sos_avg_opp_elo`, `away_sos_avg_opp_elo` | float | Computed | Strength of schedule: average opponent Elo |
| 73 | `tournament_draw_rate` | float | Computed | Historical draw rate for the tournament type |
| 74 | `elo_delta_x_home_advantage` | float | Interaction | Elo delta * home advantage |
| 75 | `fifa_rank_delta_x_same_confed` | float | Interaction | Rank delta * same confederation |
| 76-78 | `odds_home/draw/away_implied_prob` | float | Bookmakers | Implied probabilities from odds (WC2026 only) |
| 79 | `home_squad_value_m` | float | Transfermarkt | Home team total squad market value (€M) |
| 80 | `away_squad_value_m` | float | Transfermarkt | Away team total squad market value (€M) |
| 81 | `squad_value_delta` | float | Derived | Squad value difference (home - away, €M) |
| 82 | `squad_value_abs_delta` | float | Derived | Absolute squad value difference (€M) |
| 83 | `home_avg_player_value_m` | float | Transfermarkt | Home team average player market value (€M) |
| 84 | `away_avg_player_value_m` | float | Transfermarkt | Away team average player market value (€M) |
| 85 | `home_top_player_value_m` | float | Transfermarkt | Home team most valuable player market value (€M) |
| 86 | `away_top_player_value_m` | float | Transfermarkt | Away team most valuable player market value (€M) |

Note: Feature numbering in the optional section reflects the total count (27 core + 53 optional = 80 features used when all are present). The actual count available depends on data completeness. Historical matches use confederation-average fallbacks for squad quality (66% have actual data, 34% use confed avg); WC 2026 matches all have actual data (0 NaN).

## Elo Rating System

The Elo rating system in `src/features/elo.py` adapts the standard chess Elo for football.

### Core Formula

For each match, the expected score of the home team is:

```
E_home = 1 / (1 + 10^((R_away - R_home_adj) / 400))
```

Where `R_home_adj = R_home + home_advantage` if not neutral.

### Rating Update

After each match, ratings are updated:

```
R_home_new = R_home + K * (S_home - E_home)
R_away_new = R_away + K * (S_away - (1 - E_home))
```

Where:
- `S_home` = 1.0 for home win, 0.0 for away win, 0.5 for draw
- `K` = tournament-specific K-factor

### K-Factors

| Tournament Type | K-Factor |
|----------------|----------|
| FIFA World Cup | 80 |
| World Cup Qualification | 60 |
| Friendly | 40 |
| Default (other) | 50 |

Higher K-factors for World Cup matches mean ratings change more rapidly during the most important tournaments.

### Draw Probability

Draw probability is estimated as:

```
draw_prob_raw = draw_factor * (1 - |E_home - E_away|)
draw_prob = min(draw_prob, 0.35, max(0.05, min(E_home, E_away)))
remaining = 1 - draw_prob
home_win_prob = remaining * E_home_adj / (E_home_adj + E_away)
away_win_prob = remaining * E_away / (E_home_adj + E_away)
```

With `ELO_DRAW_FACTOR = 0.30` (increased from 0.25), draws are more likely between evenly-matched teams and are bounded between 5% and 35%.

### Home Advantage

Home teams receive a 100-point Elo bonus when the match is not on neutral ground. This is controlled by `ELO_HOME_ADVANTAGE = 100` in `config.py`.

## Tournament Simulation Algorithm

### Simulation Model

The simulation uses XGBoost as the default model (fast inference, 6.6 MB), loaded from `data/processed/models/xgboost.joblib`. The trained `SimpleImputer` (median strategy) is loaded from `data/processed/models/imputer.joblib` and applied to handle missing features before prediction.

### Probability Handling (Critical)

The model's `predict_proba()` method returns probabilities in class order `[away_win(0), draw(1), home_win(2)]`. Both simulators reorder this to `[home_win, draw, away_win]` before use:

```python
# In group_stage.py and knockout_stage.py:
proba = self.model.predict_proba(X)[0]
if len(proba) == 3:
    proba = proba / proba.sum()  # Normalize
    return np.array([proba[2], proba[1], proba[0]])  # [home_win, draw, away_win]
```

Probabilities are also normalized (`probs / probs.sum()`) before sampling to prevent `ValueError: Probabilities do not sum to 1`.

### Fixture Swap Handling

When a fixture lookup falls back to the reversed match (e.g., looking up "Qatar vs Switzerland" but finding "Switzerland vs Qatar"), the model's probabilities are from the wrong perspective. This is handled by adding a `swapped` flag and swapping the probability array:

```python
# If the fixture was found as away vs home (reversed), swap probabilities:
if swapped:
    probs = np.array([probs[2], probs[1], probs[0]])  # [away_win, draw, home_win] -> [home_win, draw, away_win]
```

This fix is applied in 4 locations: `group_stage.py:_predict_match()`, `knockout_stage.py:predict_knockout_match()`, `simulator.py:_predict_match_proba()`, and `live_validation.py:validate_against_live()`.

### Draw Calibration (WC Group Stage)

Historical World Cup group-stage matches have a ~25% draw rate, but the model may under-predict draws. The `_calibrate_draw()` method (in `group_stage.py` as a static method, and in `simulator.py` as a module-level function) adjusts predicted probabilities when the model's draw probability is below the expected rate:

```python
WC_GROUP_DRAW_RATE = 0.25  # Historical WC group-stage draw rate

def _calibrate_draw(probs):
    """Boost draw probability toward historical ~25% WC group-stage rate."""
    home_win_prob, draw_prob, away_win_prob = probs
    if draw_prob < WC_GROUP_DRAW_RATE:
        deficit = WC_GROUP_DRAW_RATE - draw_prob
        draw_prob = WC_GROUP_DRAW_RATE
        # Redistribute excess proportionally from win/loss probs
        total_win_loss = home_win_prob + away_win_prob
        if total_win_loss > 0:
            home_win_prob *= (1 - deficit) / total_win_loss * (home_win_prob / total_win_loss)
            # Simplified: scale win/loss proportionally
    return np.array([home_win_prob, draw_prob, away_win_prob])
```

Draw calibration is applied in group stage simulations (`_predict_match()` in `group_stage.py` and `_predict_match_proba()` in `simulator.py`) but **not** in knockout stage simulations, where draws are redistributed to produce a winner.

### Group Stage

The `GroupStageSimulator` in `src/simulation/group_stage.py` simulates each group independently:

1. **Feature preparation**: Load match features; impute NaN values using trained `SimpleImputer`
2. **Match prediction**: For each pair of teams in a group, predict win/draw/loss probabilities using the XGBoost model; reorder and normalize probabilities
3. **Outcome sampling**: Sample outcome from the predicted probability distribution
4. **Goal generation**: Generate goal totals using Poisson distributions (mean 1.5 for winners, 1.3 for away winners, 1.0 for draws)
5. **Standing calculation**: Rank teams by points (3 for win, 1 for draw, 0 for loss), then goal difference, then goals scored
6. **Advancement**: Top 2 from each group advance; best 8 third-place teams determined by sorting all third-place teams by points, then goal difference, then goals scored

### Knockout Stage

The `KnockoutStageSimulator` in `src/simulation/knockout_stage.py` simulates the knockout bracket:

1. **Feature preparation**: Same imputer and probability reordering as group stage
2. **Draw redistribution**: In knockout matches, draw probability is reduced by 70%. The redistributed probability is split 55/45 in favor of the stronger team. With the corrected probability ordering: `probs[0]` = home_win, `probs[1]` = draw, `probs[2]` = away_win
3. **Winner determination**: A single winner is sampled from the adjusted probabilities
4. **Bracket construction**: Groups A-L are mapped to bracket positions; 8 best third-place teams fill remaining R32 slots

### Full Tournament

The `WorldCupSimulator` in `src/simulation/simulator.py` runs the complete pipeline N times:

1. Load model (XGBoost by default), features, groups, and fixtures
2. For each simulation:
   a. Simulate all 12 groups
   b. Determine third-place qualifiers
   c. Build and simulate knockout bracket
   d. Track how far each team advances
3. Aggregate: compute `prob_ro32`, `prob_ro16`, `prob_qf`, `prob_sf`, `prob_final`, `prob_winner` for each team
4. If live results exist, update group standings with actual scores before knockout simulation

### 2026 Format

- 48 teams in 12 groups (A-L)
- Each group has 4 teams playing 6 round-robin matches (72 total group stage matches)
- 24 teams advance from groups (top 2 per group = 24)
- 8 best third-place teams advance (total = 32)
- Round of 32 > Round of 16 > Quarter-finals > Semi-finals > Final

## Bug Fixes and Optimizations

### Critical Bug Fixes

1. **Probability mapping in simulation**: `predict_proba()` returns `[away_win(0), draw(1), home_win(2)]` but both simulators treated it as `[home_win, draw, away_win]`. This caused inverted home/away predictions. Fixed by reordering: `return np.array([proba[2], proba[1], proba[0]])` in both `group_stage.py` and `knockout_stage.py`.

2. **Missing imputer in simulation**: Both `group_stage.py` and `knockout_stage.py` used `np.nanmean()` (global mean) to fill NaN features instead of the trained `SimpleImputer` (per-feature medians). This could produce incorrect feature values for matches with missing data. Fixed by loading `imputer.joblib` and using `imputer.transform()`.

3. **Probability normalization**: `predict_proba()` output may not sum to exactly 1.0 due to floating-point precision. Added `probs = probs / probs.sum()` before sampling in both simulators to prevent `ValueError: Probabilities do not sum to 1`.

4. **Knockout draw redistribution**: After the probability reorder fix, the draw redistribution code needed updated index mapping. `probs[0]` is now home_win, `probs[1]` is draw, `probs[2]` is away_win. The redistribution now correctly adds draw probability to home and away win probabilities.

5. **`features_list.append(features)` missing**: The feature building loop in both `build_features.py` and `build_2026_features.py` created feature dicts but never appended them to the list, causing empty DataFrames and `KeyError: 'elo_delta'`. Fixed by ensuring `features_list.append(features)` is present.

6. **`neutral=0` for all WC 2026 matches**: All 72 WC 2026 fixtures had `neutral=0`, meaning home advantage was wrongly applied to neutral-venue matches. Fixed: matches at neutral venues now get `neutral=1`; host nation home matches get `neutral=0` with `home_advantage=1.0`; away-team host matches get `home_advantage=0.3`.

7. **Stale Elo cache**: `elo_ratings_current.parquet` only had 4 teams. `build_2026_features.py` now recomputes Elo from full match history if the cache has fewer than 10 teams.

8. **WC fixture duplicates**: Fixture generation used `permutations`, creating 144 matches (12 per group). Fixed to use `combinations` for the correct 72 matches (6 per group x 12 groups).

9. **`is_knockout` barely triggered**: Only 3 out of 49,413 matches had `is_knockout=1`. Added date-based WC knockout detection (tournament start date + 15 days). Now 271 matches are correctly flagged as knockout.

10. **Fixture swap bug (CRITICAL)**: When a fixture lookup falls back to the reversed match (e.g., looking up "Qatar vs Switzerland" but finding "Switzerland vs Qatar"), the model's probabilities were from the wrong perspective — `[home_win, draw, away_win]` was actually `[away_win, draw, home_win]`. Fixed by adding a `swapped` flag and swapping the probability array `[home_win, draw, away_win]` → `[away_win, draw, home_win]` in 4 files: `group_stage.py`, `knockout_stage.py`, `simulator.py`, and `live_validation.py`.

### Model Changes

| Change | Before | After |
|--------|--------|-------|
| LightGBM | Included in ensemble | Removed (0.49 accuracy, 0.99 log_loss) |
| Best ensemble | Voting/Stacking selection | StackingClassifier (4 base + LogReg meta, class_weight="balanced") |
| Calibration | CalibratedWrapper (isotonic) | Removed (log_loss 0.8374 -> 1.0436) |
| Simulation model | Best ensemble | XGBoost (fast, 6.6 MB) |
| Draw sample weight (XGBoost, NeuralNet) | 1.5x | 4.0x |
| Stacking meta-learner class_weight | None | "balanced" |
| ELO_DRAW_FACTOR | 0.25 | 0.30 |
| Feature cache version | "1" | "7" |
| Squad quality features | Not present | 8 new features (squad value, avg/top player value) |
| WC draw calibration | Not present | `_calibrate_draw()` boosting draws to ~25% in group stage |
| Fixture swap bug | Probabilities from wrong perspective | `swapped` flag + probability swap in 4 files |

### Model Optimizations

| Change | Before | After |
|--------|--------|-------|
| RF class_weight | None | `"balanced"` |
| LogReg class_weight | None | `"balanced"` + C tuning |
| Ensemble selection metric | accuracy | log_loss |
| XGBoost Optuna scoring | accuracy | log_loss |
| RF GridSearch scoring | accuracy | neg_log_loss |
| Neural Net alpha | 0.0001 | 0.001 |
| Neural Net batch_size | 64 | 256 |
| Neural Net learning rate | fixed | adaptive |
| Neural Net max_iter | 200 | 300 |

### Model Performance (test set)

| Model | Accuracy | Log Loss | Avg Brier |
|-------|----------|----------|-----------|
| BestEnsemble (Stacking) | 0.620 | 0.836 | — |
| RandomForest | 0.606 | 0.859 | — |
| XGBoost | 0.552 | 0.918 | — |
| LogisticRegression | 0.570 | 0.878 | — |
| NeuralNet | 0.597 | 1.172 | — |

### Evaluation Improvement (2022 WC, 64 matches)

| Metric   | Before  | After   |
|----------|---------|---------|
| Accuracy | 56.25%  | 59.38%  |
| Log Loss | 0.9789  | 0.9383  |
| Avg Brier| 0.1913  | 0.1657  |

## Live Validation Flow

```
┌─────────────────────┐
│ scrape_live_results  │──── ESPN/Wikipedia ──> data/raw/wc2026_results_live.csv
└─────────┬───────────┘
          │
          v
┌─────────────────────┐     ┌──────────────────────────┐
│ Manual override CSV  │────>│  Merge (manual takes      │
│ (wc2026_results_     │     │  priority for same match)  │
│  manual.csv)          │     └────────────┬─────────────┘
└──────────────────────┘                  │
                                          v
                             ┌─────────────────────────┐
                             │  validate_against_live()  │
                             │  - Load best model        │
                             │  - Load WC2026 features   │
                             │  - For each played match: │
                             │    - Predict probabilities│
                             │    - Compare to actual    │
                             │    - Compute accuracy     │
                             │    - Compute log loss     │
                             └────────────┬──────────────┘
                                          │
                                          v
                             ┌─────────────────────────┐
                             │ live_validation_report   │
                             │ .csv (per-match details) │
                             │ + console summary        │
                             └─────────────────────────┘
```

When `--retrain` is passed, live match results are also included in the training data, allowing the model to learn from tournament outcomes as they happen.