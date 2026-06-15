import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestModelTraining:
    def test_model_predictions_shape(self, trained_model, sample_data):
        X, y = sample_data
        preds = trained_model.predict(X)
        assert preds.shape == (200,)

    def test_model_probability_shape(self, trained_model, sample_data):
        X, y = sample_data
        probs = trained_model.predict_proba(X)
        assert probs.shape == (200, 3)

    def test_probability_sums_to_one(self, trained_model, sample_data):
        X, y = sample_data
        probs = trained_model.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01)

    def test_probability_bounds(self, trained_model, sample_data):
        X, y = sample_data
        probs = trained_model.predict_proba(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestLabelMapping:
    def test_label_map(self):
        from src.models.train import LABEL_MAP

        assert LABEL_MAP[1] == 2
        assert LABEL_MAP[0] == 1
        assert LABEL_MAP[-1] == 0

    def test_label_names(self):
        from src.models.train import LABEL_NAMES

        assert LABEL_NAMES[0] == "away_win"
        assert LABEL_NAMES[1] == "draw"
        assert LABEL_NAMES[2] == "home_win"


class TestEnsemble:
    def test_voting_ensemble(self, sample_data):
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier

        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)

        voting = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf)], voting="soft"
        )
        voting.fit(X, y)

        preds = voting.predict(X)
        probs = voting.predict_proba(X)

        assert preds.shape == (200,)
        assert probs.shape == (200, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01)