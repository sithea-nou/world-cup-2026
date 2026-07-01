# World Cup 2026 ML Predictor

## Commands

- **Install**: `uv sync` (add `--group dev` for pytest/ruff)
- **Lint**: `uv run ruff check .` and `uv run ruff format --check .`
- **Test all**: `uv run pytest`
- **Test single file**: `uv run pytest tests/test_elo.py`
- **Test single class/method**: `uv run pytest tests/test_models.py::TestModelTraining -v`
- **Run pipeline**: `python run_pipeline.py --all` (full), `python run_pipeline.py --step scraping` (single step)

## Pipeline step order matters

`scraping -> features -> train -> ensemble -> evaluate -> simulate -> visualize`

Each step depends on outputs from prior steps (files in `data/raw/` then `data/processed/`). Running steps out of order will fail with missing data errors.

## Architecture

- `run_pipeline.py` — CLI orchestrator, single entrypoint
- `src/config.py` — all constants (Elo params, draw rates, K-factors, team name mappings); `OPTUNA_TRIALS=20`, `CV_FOLDS=3`
- `src/helpers.py` — logging, Kaggle setup, `normalize_team_name()`
- `src/scraping/` — data acquisition (Kaggle API, Wikipedia, ESPN, the-odds-api, Transfermarkt)
- `src/features/` — Elo rating system (`elo.py`), feature engineering (`build_features.py`, cache versioned via hash)
- `src/models/` — training (`train.py`), ensemble selection (`ensemble.py`: Voting, WeightedVoting, Stacking), evaluation, live validation, prediction helpers
- `src/models/prediction.py` — `predict_with_draw_threshold()` and `predict_proba_with_draw_boost()` for draw-aware predictions
- `src/simulation/` — Monte Carlo tournament sim (group stage, knockout, orchestrator); uses XGBoost for speed
- `src/visualization/` — plots and tables

## Critical implementation details

- **Label encoding**: `away_win=0, draw=1, home_win=2`. `predict_proba()` returns `[away_win, draw, home_win]` — must reorder to `[home_win, draw, away_win]` for simulation
- **Draw handling**: XGBoost uses 4x sample weight for draws (default), NeuralNet uses 8x; RF/LogReg use `class_weight="balanced"`; meta-learner does NOT use class_weight
- **Draw calibration**: `WC_GROUP_DRAW_RATE=0.25` boosts under-predicted draws in group stage; NOT applied in knockout
- **Fixture swap bug**: when match lookup is reversed, probabilities must be swapped `[home_win, draw, away_win] -> [away_win, draw, home_win]` — already fixed in 4 files
- **Imputer**: simulation must load `imputer.joblib` (per-feature medians), NOT `np.nanmean()`
- **ELO_DRAW_FACTOR=0.30**, **ELO_HOME_ADVANTAGE=100**
- **Kernel issue**: models saved with NumPy 2.4.6 (Python 3.13/uv) cannot be loaded by NumPy 1.26.4 (Python 3.12/Anaconda); Jupyter kernel registered as `worldcup-2026-predictor`
- **torch.nn removal**: `import torch.nn as nn` was removed from train.py — it loaded ~1GB unused PyTorch, causing kernel death when combined with RF+StackingClassifier memory
- **Model sizes**: best_model.joblib=96MB, randomforest.joblib=47MB, xgboost.joblib=819KB (compressed with `joblib.dump(..., compress=3)`)
- **Simulation double-counting fix**: `_apply_live_results()` in simulator now adds only real results; `simulate_group()` skips played matches via `played_matches` set to avoid double-counting
- **Draw prediction**: model well-calibrated but argmax rarely picks draw; 5 actual draws in WC2026 all missed (Canada-BIH, Qatar-SUI, Brazil-MAR, Netherlands-JPN, Spain-CPV); `draw_threshold` doesn't help with current matches
- **Best model**: WeightedVotingEnsemble (inverse-log-loss weighted soft voting across 4 base models), selected by validation log_loss over VotingEnsemble and StackingEnsemble candidates

## Environment

- **Python 3.13+**, managed by `uv`
- **Kaggle API**: requires `~/.kaggle/kaggle.json`
- **ODDS_API_KEY**: optional env var for the-odds-api outright odds
- `data/raw/` and `data/processed/` are gitignored; `data/external/continents.csv` is gitignored but shipped

## Style

- Ruff: line-length 100, target py313, rules `E F W I`
- No LightGBM in training (worst performer, function preserved but not called)
- No calibration wrapper (hurt log_loss: 0.84 -> 1.04, code preserved but not used)