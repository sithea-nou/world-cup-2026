# Contributing Guide

Thank you for your interest in contributing to the World Cup 2026 ML Predictor. This document describes how to set up the development environment, follow code style conventions, write tests, and submit contributions.

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd worldcup

# Install all dependencies including dev tools
uv sync --group dev
```

This installs `pytest`, `ruff`, and all runtime dependencies defined in `pyproject.toml`.

### Verify Installation

```bash
# Run the test suite
uv run pytest

# Run a quick pipeline check
python run_pipeline.py --setup-only
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. The configuration is in `pyproject.toml`.

### Rules

- **Line length**: 100 characters maximum
- **Target**: Python 3.13
- **Lint rules**: E (errors), F (pyflakes), W (warnings), I (isort)
- **No comments in code** unless explicitly requested. Code should be self-documenting through clear naming and structure.
- **No emojis** in code or documentation unless explicitly requested.

### Formatting and Linting

```bash
# Check for linting errors
uv run ruff check .

# Auto-fix linting errors
uv run ruff check --fix .

# Check formatting
uv run ruff format --check .

# Auto-format
uv run ruff format .
```

### Naming Conventions

- **Modules**: `snake_case.py` (e.g., `build_features.py`, `group_stage.py`)
- **Classes**: `PascalCase` (e.g., `EloRatingSystem`, `GroupStageSimulator`, `WorldCupSimulator`)
- **Functions**: `snake_case` (e.g., `build_match_features`, `train_xgboost`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RANDOM_STATE`, `N_SIMULATIONS`)
- **Private functions**: Prefix with `_` (e.g., `_get_fifa_ranking`, `_compute_form_features`)

### Import Order

Imports are organized by ruff's isort rules:

1. Standard library
2. Third-party packages
3. Local application imports (`from src.`)

Example:

```python
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, PROCESSED_DIR
from src.helpers import logger
```

## Testing Guidelines

The project uses `pytest` with tests across 4 test files.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_elo.py

# Run a specific test class
uv run pytest tests/test_models.py::TestModelTraining

# Run a specific test
uv run pytest tests/test_elo.py::TestEloProbability::test_probability_bounds

# Run with verbose output
uv run pytest -v

# Run with print output
uv run pytest -s
```

### Test Structure

Tests are organized in `tests/`:

```
tests/
├── conftest.py         # Shared fixtures (sample_match_data, sample_groups, tmp_data_dir)
├── test_elo.py         # Elo rating system tests
├── test_features.py    # Feature engineering tests
├── test_models.py      # Model training and ensemble tests
└── test_simulation.py  # Simulation and tournament tests
```

### Writing Tests

Follow these conventions:

1. **Group tests in classes** by functional area:

```python
class TestEloComputation:
    def test_compute_elo_ratings(self, elo_system, sample_matches):
        ...

    def test_elo_changes_after_matches(self, elo_system, sample_matches):
        ...
```

2. **Use fixtures** for shared test data:

```python
@pytest.fixture
def sample_matches():
    return pd.DataFrame([...])

@pytest.fixture
def elo_system():
    return EloRatingSystem()
```

3. **Test behavior, not implementation**:

```python
# Good: test the public interface
def test_probability_bounds(self, elo_system):
    probs = elo_system.predict_match_probability(1500, 1500, neutral=True)
    assert 0 < probs["home_win"] < 1
    assert abs(sum(probs.values()) - 1.0) < 0.01

# Avoid: testing private implementation details
```

4. **Use descriptive test names** that explain the expected behavior:

```python
def test_higher_elo_more_likely_to_win(self, elo_system):
    ...
```

5. **Keep tests fast**: Use mock models and small datasets. Avoid tests that require downloading data or training on the full dataset.

### Mock Models

For simulation tests, use simple mock models. Note that `predict_proba()` returns `[away_win, draw, home_win]` (class order 0, 1, 2):

```python
class MockModel:
    def predict_proba(self, X):
        # Returns [away_win, draw, home_win] (class order 0, 1, 2)
        return np.array([[0.3, 0.2, 0.5]])

    def predict(self, X):
        return np.array([2])
```

### Testing Probability Reordering

The simulation code reorders `predict_proba()` output from `[away_win, draw, home_win]` to `[home_win, draw, away_win]`. Tests should verify this reordering:

```python
def test_predict_proba_reordering(self):
    """Verify _predict_match reorders probabilities correctly."""
    # predict_proba returns [away_win(0), draw(1), home_win(2)]
    # _predict_match should return [home_win, draw, away_win]
    probs = simulator._predict_match("Brazil", "Argentina", wc_features)
    assert probs[0] == home_win_prob   # was proba[2]
    assert probs[1] == draw_prob       # was proba[1]
    assert probs[2] == away_win_prob   # was proba[0]
```

### Testing Imputer Usage

Simulation tests should verify that NaN features are handled by the trained imputer, not `np.nanmean()`:

```python
def test_nan_features_imputed_correctly(self):
    """NaN features should use per-feature median imputation, not global mean."""
    features_with_nan = ...  # features with NaN values
    result = simulator._predict_match("Team A", "Team B", features_with_nan)
    # Should not raise ValueError or produce nonsensical results
    assert result is not None
```

## Adding New Data Sources

### 1. Create a Scraping Module

Add a new file in `src/scraping/` (e.g., `src/scraping/scrape_new_source.py`):

```python
import pandas as pd
from src.config import RAW_DIR
from src.helpers import logger


def scrape_new_source() -> pd.DataFrame:
    """Scrape data from new source.

    Returns:
        DataFrame with scraped data
    """
    logger.info("Scraping from new source...")

    # Implement scraping logic here
    # ...

    out_path = RAW_DIR / "new_source_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} records to {out_path}")

    return df
```

### 2. Add Configuration

Add any URLs, API keys, or constants to `src/config.py`:

```python
NEW_SOURCE_URL = "https://example.com/api"
NEW_SOURCE_API_KEY = os.environ.get("NEW_SOURCE_API_KEY", "")
```

If the new data affects feature engineering, also increment `FEATURES_CACHE_VERSION` in `src/features/build_features.py`.

### 3. Integrate into Pipeline

Add the scraping call to `run_scraping()` in `run_pipeline.py`:

```python
def run_scraping():
    ...
    from src.scraping.scrape_new_source import scrape_new_source
    scrape_new_source()
    ...
```

### 4. Use in Feature Engineering

Load the new data in `src/features/build_features.py` (or `build_2026_features.py`):

```python
def load_all_data(include_live: bool = False) -> dict[str, pd.DataFrame]:
    ...
    new_path = RAW_DIR / "new_source_data.csv"
    if new_path.exists():
        data["new_source"] = pd.read_csv(new_path)
        logger.info(f"  New source: {len(data['new_source'])} rows")
    ...
```

### 5. Add Tests

Add basic tests for the new source:

```python
def test_scrape_new_source_returns_dataframe():
    ...
```

## Adding New Models

### 1. Create a Training Function

Add a training function in `src/models/train.py` following the existing pattern:

```python
def train_new_model(X_train, y_train, X_val=None, y_val=None) -> dict:
    """Train a new model.

    Returns:
        Dict with keys 'model', 'params', 'name'
    """
    logger.info("Training New Model...")

    # Implement training logic
    model = NewClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "params": {"param1": value1},
        "name": "NewModel",
    }
```

### 2. Add to Pipeline

Add the new model to `run_train()` in `run_pipeline.py`:

```python
def run_train(include_live: bool = False):
    ...
    from src.models.train import train_new_model

    logger.info("Training New Model...")
    new_model = train_new_model(X_train, y_train, X_val, y_val)

    models["NewModel"] = new_model
    ...
```

### 3. Ensure Compatibility

The new model must implement:
- `fit(X, y)` -- for training
- `predict(X)` -- returns class labels
- `predict_proba(X)` -- returns probability matrix (N x 3)

**Important**: `predict_proba()` returns probabilities in class order `[away_win(0), draw(1), home_win(2)]`. The simulation code reorders this to `[home_win, draw, away_win]`. If your model outputs a different class order, you must update the simulation code accordingly.

If the model does not support `predict_proba`, it will be excluded from the stacking ensemble automatically.

### 4. Add Feature Columns

If the model requires specific features, add them to `FEATURE_COLUMNS` or `OPTIONAL_FEATURES` in `src/models/train.py`. When adding or modifying features, you must also increment `FEATURES_CACHE_VERSION` in `src/features/build_features.py` to invalidate stale caches. The current version is "7".

### 5. Ensemble Selection

The ensemble builder (`build_best_ensemble` in `src/models/ensemble.py`) selects models by validation **log_loss** (not accuracy). If your new model is added to `run_train()`, it will automatically be considered as an ensemble candidate. Make sure it implements `predict_proba()`.

### 6. Add Tests

```python
class TestNewModel:
    def test_model_predictions_shape(self):
        ...

    def test_model_probability_shape(self):
        ...

    def test_predict_proba_class_order(self):
        # Verify predict_proba returns [away_win, draw, home_win]
        probs = model.predict_proba(X_test)
        assert probs.shape[1] == 3
        # Classes should be 0=away_win, 1=draw, 2=home_win
```

## Key Implementation Notes

### Probability Class Ordering

scikit-learn models return `predict_proba()` output in class order (0, 1, 2), which maps to `{0: away_win, 1: draw, 2: home_win}`. The simulation code in `group_stage.py` and `knockout_stage.py` reorders this to `[home_win, draw, away_win]` before use. When adding new simulation code, ensure this reordering is applied:

```python
proba = self.model.predict_proba(X)[0]
if len(proba) == 3:
    proba = proba / proba.sum()  # Normalize
    return np.array([proba[2], proba[1], proba[0]])  # [home_win, draw, away_win]
```

### Imputer Usage

The simulation code uses a trained `SimpleImputer` (median strategy) for handling NaN features. This is loaded from `data/processed/models/imputer.joblib`. Do not use `np.nanmean()` for imputation in simulation code, as it produces a global mean rather than per-feature medians.

### Probability Normalization

Always normalize probabilities before sampling with `np.random.choice`:

```python
probs = probs / probs.sum()  # Prevent ValueError: Probabilities do not sum to 1
outcome = self.rng.choice(["home_win", "draw", "away_win"], p=probs)
```

### Feature Cache Versioning

When modifying features, always increment `FEATURES_CACHE_VERSION` in `src/features/build_features.py`. The current version is "7". This ensures stale caches are rebuilt with the new feature definitions.

### Simulation Model

The default simulation model is XGBoost (loaded from `xgboost.joblib`), not the best ensemble. This is because XGBoost is faster for inference (819KB compressed). The `WorldCupSimulator` class accepts a `model_name` parameter to specify which model file to load. Model files are compressed with `joblib.dump(..., compress=3)`; best_model.joblib is 96MB, randomforest.joblib is 47MB.

## PR Process

### Before Submitting

1. **Run the full test suite** and ensure all tests pass:

```bash
uv run pytest
```

2. **Run the linter** and fix any errors:

```bash
uv run ruff check .
uv run ruff format .
```

3. **Test the pipeline end-to-end** on a small dataset if possible.

### Pull Request Template

PRs should include:

- **Description**: What the PR does and why
- **Changes**: List of modified files and what changed
- **Testing**: How the changes were tested
- **Breaking changes**: Any changes that affect existing functionality

### Review Criteria

- All tests pass
- No linting errors
- New code has tests
- New data sources include documentation in `docs/DATA_SOURCES.md`
- New models include documentation in `docs/ARCHITECTURE.md` and `docs/API.md`
- Feature changes increment `FEATURES_CACHE_VERSION` in `src/features/build_features.py`
- `predict_proba()` class ordering is handled correctly in simulation code
- Imputer usage follows the trained `SimpleImputer` pattern (not `np.nanmean()`)
- Probabilities are normalized before sampling
- No unnecessary comments in code
- No emojis in code or documentation