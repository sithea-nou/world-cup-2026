import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import torch
import torch.nn as nn

from src.config import (
    CV_FOLDS,
    NEURAL_NET_DROPOUT,
    NEURAL_NET_EPOCHS,
    NEURAL_NET_LAYERS,
    NEURAL_NET_LEARNING_RATE,
    NEURAL_NET_PATIENCE,
    OPTUNA_TRIALS,
    PROCESSED_DIR,
    RANDOM_STATE,
)
from src.helpers import logger

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLUMNS = [
    "elo_home",
    "elo_away",
    "elo_delta",
    "elo_abs_delta",
    "elo_home_win_prob",
    "elo_draw_prob",
    "elo_away_win_prob",
    "fifa_rank_home",
    "fifa_rank_away",
    "fifa_rank_delta",
    "fifa_rank_abs_delta",
    "fifa_points_home",
    "fifa_points_away",
    "fifa_points_delta",
    "fifa_points_abs_delta",
    "neutral",
    "home_advantage",
    "is_host_nation",
    "same_confederation",
    "is_world_cup",
    "is_qualifier",
    "is_friendly",
    "is_knockout",
    "combined_draw_prob",
    "elo_close",
    "draw_tendency",
    "fifa_close",
]

OPTIONAL_FEATURES = [
    "home_form_last10_win_rate",
    "home_form_last10_draw_rate",
    "home_form_last10_loss_rate",
    "home_form_last10_goals_scored_avg",
    "home_form_last10_goals_conceded_avg",
    "home_form_last10_goal_diff_avg",
    "home_form_last10_clean_sheet_rate",
    "home_form_last5_win_rate",
    "home_form_last5_draw_rate",
    "home_form_last5_loss_rate",
    "home_form_last5_goals_scored_avg",
    "home_form_last5_goals_conceded_avg",
    "home_form_last5_goal_diff_avg",
    "home_form_last5_clean_sheet_rate",
    "away_form_last10_win_rate",
    "away_form_last10_draw_rate",
    "away_form_last10_loss_rate",
    "away_form_last10_goals_scored_avg",
    "away_form_last10_goals_conceded_avg",
    "away_form_last10_goal_diff_avg",
    "away_form_last10_clean_sheet_rate",
    "away_form_last5_win_rate",
    "away_form_last5_draw_rate",
    "away_form_last5_loss_rate",
    "away_form_last5_goals_scored_avg",
    "away_form_last5_goals_conceded_avg",
    "away_form_last5_goal_diff_avg",
    "away_form_last5_clean_sheet_rate",
    "away_form_last10_ewm_win_rate",
    "away_form_last10_ewm_draw_rate",
    "away_form_last10_ewm_loss_rate",
    "away_form_last5_ewm_win_rate",
    "away_form_last5_ewm_draw_rate",
    "away_form_last5_ewm_loss_rate",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_draw_rate",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "elo_delta_x_home_advantage",
    "fifa_rank_delta_x_same_confed",
    "home_sos_avg_opp_elo",
    "away_sos_avg_opp_elo",
    "tournament_draw_rate",
    "odds_home_implied_prob",
    "odds_draw_implied_prob",
    "odds_away_implied_prob",
    "home_squad_value_m",
    "away_squad_value_m",
    "squad_value_delta",
    "squad_value_abs_delta",
    "home_avg_player_value_m",
    "away_avg_player_value_m",
    "home_top_player_value_m",
    "away_top_player_value_m",
]

TARGET_COLUMN = "outcome"
LABEL_MAP = {1: 2, 0: 1, -1: 0}
LABEL_NAMES = {0: "away_win", 1: "draw", 2: "home_win"}


def _build_sklearn_mlp():
    """Build a sklearn MLPClassifier as the neural net model."""
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=tuple(NEURAL_NET_LAYERS),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=NEURAL_NET_LEARNING_RATE,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=NEURAL_NET_PATIENCE,
        random_state=RANDOM_STATE,
        tol=1e-4,
    )


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    available = []
    for col in FEATURE_COLUMNS + OPTIONAL_FEATURES:
        if col in df.columns:
            available.append(col)
    return available


def split_data(
    df: pd.DataFrame, test_year: int = 2022, val_years: list[int] | None = None
) -> tuple:
    logger.info("Splitting data...")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if val_years is None:
        val_years = [2022]

    train_mask = df["date"].dt.year < min(val_years)
    val_mask = df["date"].dt.year.isin(val_years)
    test_mask = df["date"].dt.year > max(val_years)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    feature_cols = _get_feature_columns(df)

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].map(LABEL_MAP).values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COLUMN].map(LABEL_MAP).values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COLUMN].map(LABEL_MAP).values

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = np.ascontiguousarray(imputer.fit_transform(X_train), dtype=np.float32)
    X_val = np.ascontiguousarray(imputer.transform(X_val), dtype=np.float32)
    X_test = np.ascontiguousarray(imputer.transform(X_test), dtype=np.float32)

    import joblib
    imputer_path = PROCESSED_DIR / "models" / "imputer.joblib"
    imputer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, imputer_path)

    return (
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_cols, train_df, val_df, test_df,
    )


def _compute_sample_weights(y_train):
    from sklearn.utils.class_weight import compute_sample_weight
    class_weights = {0: 1.0, 1: 4.0, 2: 1.0}
    return compute_sample_weight(class_weights, y_train)


def train_xgboost(X_train, y_train, X_val=None, y_val=None) -> dict:
    logger.info("Training XGBoost...")

    sample_weights = _compute_sample_weights(y_train)

    try:
        import optuna
        from optuna.samplers import TPESampler

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            }

            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )

            if X_val is not None and y_val is not None:
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                from sklearn.metrics import log_loss
                y_prob = model.predict_proba(X_val)
                score = log_loss(y_val, y_prob)
            else:
                tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
                scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_log_loss")
                score = -scores.mean()

            return score

        study.optimize(objective, n_trials=min(OPTUNA_TRIALS, 30), show_progress_bar=False)
        best_params = study.best_params
        logger.info(f"XGBoost best params: {best_params}")
    except ImportError:
        logger.warning("Optuna not available, using default XGBoost params")
        best_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
        }

    model = XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    return {"model": model, "params": best_params, "name": "XGBoost"}


def train_random_forest(X_train, y_train) -> dict:
    logger.info("Training Random Forest...")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt"],
        "class_weight": ["balanced"],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_score = grid.best_score_
    best_model = grid.best_estimator_

    logger.info(f"Random Forest best params: {best_params} (CV neg_log_loss: {best_score:.4f})")
    return {"model": best_model, "params": best_params, "name": "RandomForest"}


def train_logistic_regression(X_train, y_train) -> dict:
    logger.info("Training Logistic Regression...")

    from sklearn.model_selection import GridSearchCV

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    param_grid = {"lr__C": [0.01, 0.1, 1.0, 10.0]}
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info(f"Logistic Regression best C: {grid.best_params_['lr__C']}")
    return {"model": grid.best_estimator_, "params": grid.best_params_, "name": "LogisticRegression"}


def train_lightgbm(X_train, y_train, X_val=None, y_val=None) -> dict:
    logger.info("Training LightGBM...")

    from lightgbm import LGBMClassifier

    sample_weights = _compute_sample_weights(y_train)

    model = LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_leaves=63,
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(stopping_rounds=50),
                __import__("lightgbm").log_evaluation(period=0),
            ],
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    return {"model": model, "params": {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.05}, "name": "LightGBM"}


def train_neural_net(X_train, y_train, X_val=None, y_val=None) -> dict:
    """Train a neural network classifier using sklearn's MLPClassifier."""
    logger.info("Training Neural Network (sklearn MLP)...")

    model = _build_sklearn_mlp()
    sample_weights = _compute_sample_weights(y_train)

    if X_val is not None and y_val is not None:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        sw_combined = np.concatenate([sample_weights, _compute_sample_weights(y_val)])
        model.fit(X_combined, y_combined)
    else:
        model.fit(X_train, y_train)

    return {
        "model": model,
        "params": {
            "layers": NEURAL_NET_LAYERS,
            "dropout": NEURAL_NET_DROPOUT,
            "learning_rate": NEURAL_NET_LEARNING_RATE,
        },
        "name": "NeuralNet",
    }


def save_all_models(models: dict[str, dict], feature_cols: list[str]):
    logger.info("Saving all models...")

    models_dir = PROCESSED_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model_dict in models.items():
        path = models_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model_dict["model"], path)
        logger.info(f"  Saved {name} to {path}")

    joblib.dump(feature_cols, models_dir / "feature_columns.joblib")
    logger.info(f"  Saved feature columns ({len(feature_cols)}) to {models_dir / 'feature_columns.joblib'}")


if __name__ == "__main__":
    features_path = PROCESSED_DIR / "match_features.parquet"
    if not features_path.exists():
        logger.error("Features not found. Run build_features.py first.")
    else:
        df = pd.read_parquet(features_path)
        df = df.dropna(subset=[TARGET_COLUMN])

        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, train_df, val_df, test_df,
        ) = split_data(df)

        xgb = train_xgboost(X_train, y_train, X_val, y_val)
        rf = train_random_forest(X_train, y_train)
        lr = train_logistic_regression(X_train, y_train)
        nn = train_neural_net(X_train, y_train, X_val, y_val)

        models = {
            "XGBoost": xgb,
            "RandomForest": rf,
            "LogisticRegression": lr,
            "NeuralNet": nn,
        }

        save_all_models(models, feature_cols)

        for name, model_dict in models.items():
            model = model_dict["model"]
            if hasattr(model, "score"):
                train_acc = model.score(X_train, y_train)
                val_acc = model.score(X_val, y_val) if X_val is not None else None
                logger.info(f"{name}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f if val_acc else 'N/A'}")