import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

from src.config import PROCESSED_DIR, RANDOM_STATE
from src.helpers import logger

warnings.filterwarnings("ignore", category=UserWarning)


class CalibratedWrapper:
    def __init__(self, base_model, isotonic_regressors=None, n_classes=3):
        self.base_model = base_model
        self.isotonic_regressors = isotonic_regressors or []
        self.n_classes = n_classes
        self.classes_ = np.array(list(range(n_classes)))

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)
        if not self.isotonic_regressors:
            return base_probs
        calibrated = np.zeros_like(base_probs)
        for cls in range(self.n_classes):
            if cls < len(self.isotonic_regressors):
                calibrated[:, cls] = self.isotonic_regressors[cls].transform(base_probs[:, cls])
            else:
                calibrated[:, cls] = base_probs[:, cls]
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


class WeightedEnsemble:
    def __init__(self, sklearn_model, nn_model, X_val=None, y_val=None):
        self.sklearn_model = sklearn_model
        self.nn_model = nn_model
        self.classes_ = np.array([0, 1, 2])
        self.nn_weight = 0.3
        self.name = "WeightedEnsemble"

        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)

    def _optimize_weights(self, X_val, y_val, max_samples=500):
        n = min(len(X_val), max_samples)
        X_sub = np.array(X_val[:n], dtype=np.float32, order="C", copy=True)
        y_sub = y_val[:n]
        sk_probs = self.sklearn_model.predict_proba(X_sub)
        nn_probs = self.nn_model.predict_proba(X_sub)
        best_acc = 0
        best_w = 0.2
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            blended = (1 - w) * sk_probs + w * nn_probs
            preds = np.argmax(blended, axis=1)
            acc = accuracy_score(y_sub, preds)
            if acc > best_acc:
                best_acc = acc
                best_w = w
        self.nn_weight = best_w
        self.score_ = best_acc

    def predict_proba(self, X):
        sk_probs = self.sklearn_model.predict_proba(X)
        nn_probs = self.nn_model.predict_proba(X)
        return (1 - self.nn_weight) * sk_probs + self.nn_weight * nn_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_params(self, deep=True):
        return {"nn_weight": self.nn_weight}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_stacking_ensemble(
    models: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    logger.info("Building stacking ensemble...")

    estimators = []
    for name, model_dict in models.items():
        model = model_dict["model"]
        if hasattr(model, "predict_proba") and hasattr(model, "fit"):
            estimators.append((name, model))
        else:
            logger.warning(f"  Skipping {name} - no predict_proba or fit method")

    if len(estimators) < 2:
        logger.warning("Not enough models for stacking ensemble, using single model")
        return models[list(models.keys())[0]] if models else None

    meta_learner = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=1,
    )

    stacking.fit(X_train, y_train)

    return {
        "model": stacking,
        "params": {"estimators": [e[0] for e in estimators]},
        "name": "StackingEnsemble",
    }


def build_voting_ensemble(
    models: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: list[float] | None = None,
) -> dict:
    logger.info("Building voting ensemble...")

    estimators = []
    for name, model_dict in models.items():
        model = model_dict["model"]
        if hasattr(model, "predict_proba") and hasattr(model, "fit"):
            estimators.append((name, model))

    if len(estimators) < 2:
        logger.warning("Not enough models for voting ensemble")
        return models[list(models.keys())[0]] if models else None

    voting = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=1,
    )

    voting.fit(X_train, y_train)

    return {"model": voting, "params": {"weights": weights}, "name": "VotingEnsemble"}


def calibrate_ensemble(
    ensemble_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> dict:
    logger.info(f"Calibrating ensemble with {method}...")

    y_val_arr = np.asarray(y_val)
    n_classes = len(np.unique(y_val_arr))
    y_val_binarized = label_binarize(y_val_arr, classes=list(range(n_classes)))

    base_probs = ensemble_model.predict_proba(X_val)

    isotonic_regressors = []
    for cls in range(n_classes):
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(base_probs[:, cls], y_val_binarized[:, cls])
        isotonic_regressors.append(iso_reg)

    calibrated_model = CalibratedWrapper(ensemble_model, isotonic_regressors, n_classes)

    return {
        "model": calibrated_model,
        "params": {"calibration_method": method},
        "name": f"CalibratedEnsemble_{method}",
    }


def _evaluate_ensemble(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs) if probs is not None else np.inf
    return {"accuracy": acc, "log_loss": ll}


def build_best_ensemble(
    models: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    logger.info("Building best ensemble across voting, stacking, and single models...")

    candidates = {}

    # Evaluate each individual model
    for name, model_dict in models.items():
        m = model_dict["model"]
        if hasattr(m, "predict"):
            candidates[name] = {"model": m, "name": name}
            metrics = _evaluate_ensemble(m, X_val, y_val)
            logger.info(
                f"  {name} val_acc={metrics['accuracy']:.4f}, log_loss={metrics['log_loss']:.4f}"
            )

    # Soft-voting ensemble with uniform weights
    voting_uniform = build_voting_ensemble(models, X_train, y_train)
    if voting_uniform is not None:
        candidates[voting_uniform["name"]] = voting_uniform
        metrics = _evaluate_ensemble(voting_uniform["model"], X_val, y_val)
        logger.info(
            f"  {voting_uniform['name']} val_acc={metrics['accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}"
        )

    # Soft-voting ensemble with inverse-log-loss weights
    try:
        val_losses = {}
        for name, model_dict in models.items():
            m = model_dict["model"]
            if hasattr(m, "predict_proba"):
                probs = m.predict_proba(X_val)
                val_losses[name] = log_loss(y_val, probs)

        if val_losses:
            inv_losses = {name: 1.0 / max(loss, 1e-6) for name, loss in val_losses.items()}
            total = sum(inv_losses.values())
            # Use inverse-log-loss weights for each base model
            weights = [inv_losses.get(name, 0.0) / total for name in models if name in inv_losses]
            if weights and len(weights) == len(models):
                voting_weighted = build_voting_ensemble(models, X_train, y_train, weights=weights)
                if voting_weighted is not None:
                    candidates["WeightedVotingEnsemble"] = voting_weighted
                    metrics = _evaluate_ensemble(voting_weighted["model"], X_val, y_val)
                    logger.info(
                        f"  {voting_weighted['name']} val_acc="
                        f"{metrics['accuracy']:.4f}, "
                        f"log_loss={metrics['log_loss']:.4f}"
                    )
    except Exception as e:
        logger.warning(f"  Weighted voting build failed: {e}")

    # Stacking ensemble (uncalibrated for speed)
    stacking = build_stacking_ensemble(models, X_train, y_train)
    if stacking is not None:
        candidates[stacking["name"]] = stacking
        metrics = _evaluate_ensemble(stacking["model"], X_val, y_val)
        logger.info(
            f"  {stacking['name']} val_acc={metrics['accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}"
        )

    # Pick best by validation log_loss (lower is better); tie-break by higher accuracy
    best_name = None
    best_ll = np.inf
    best_acc = -1.0
    for name, cand in candidates.items():
        metrics = _evaluate_ensemble(cand["model"], X_val, y_val)
        if (metrics["log_loss"] < best_ll) or (
            metrics["log_loss"] == best_ll and metrics["accuracy"] > best_acc
        ):
            best_ll = metrics["log_loss"]
            best_acc = metrics["accuracy"]
            best_name = name

    if best_name is None:
        logger.warning("No ensemble candidate succeeded, falling back to first model")
        best_name = list(models.keys())[0]

    best_model = candidates[best_name]["model"]

    logger.info(
        f"  Selected best model: {best_name} (val_acc={best_acc:.4f}, log_loss={best_ll:.4f})"
    )

    models_dir = PROCESSED_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path, compress=3)
    logger.info(f"  Best ensemble saved to {model_path}")

    return {"model": best_model, "name": best_name}


def save_best_ensemble(ensemble_model, feature_cols: list[str]):
    models_dir = PROCESSED_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    if isinstance(ensemble_model, dict):
        joblib.dump(ensemble_model["model"], model_path, compress=3)
    else:
        joblib.dump(ensemble_model, model_path, compress=3)

    feat_path = models_dir / "feature_columns.joblib"
    joblib.dump(feature_cols, feat_path, compress=3)

    logger.info(f"Best ensemble saved to {model_path}")


if __name__ == "__main__":
    models_dir = PROCESSED_DIR / "models"

    models = {}
    for model_file in models_dir.glob("*.joblib"):
        if model_file.stem == "feature_columns":
            continue
        if model_file.stem == "best_model":
            continue

        name = model_file.stem.replace("_", " ").title()
        model = joblib.load(model_file)
        models[name] = {"model": model, "name": name}
        logger.info(f"Loaded {name}")

    if not models:
        logger.error("No models found. Run train.py first.")
    else:
        features_path = PROCESSED_DIR / "match_features.parquet"
        df = pd.read_parquet(features_path)
        df = df.dropna(subset=["outcome"])

        from src.models.train import split_data

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
            train_df,
            val_df,
            test_df,
        ) = split_data(df)

        best = build_best_ensemble(models, X_train, y_train, X_val, y_val)
        logger.info(f"Best ensemble: {best['name']}")
