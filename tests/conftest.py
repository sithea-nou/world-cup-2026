import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os


@pytest.fixture
def tmp_data_dir(tmp_path):
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    return {"raw": raw, "processed": processed, "root": tmp_path}


@pytest.fixture
def sample_match_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="7D")
    teams = ["Brazil", "Argentina", "France", "Germany", "Spain", "England", "Italy", "Netherlands"]

    records = []
    for i in range(n):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        home_score = np.random.poisson(1.5)
        away_score = np.random.poisson(1.2)

        records.append(
            {
                "date": dates[i],
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": np.random.choice(["Friendly", "FIFA World Cup qualification", "FIFA World Cup"]),
                "neutral": np.random.choice([True, False]),
                "city": "Test City",
                "country": "Test Country",
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def sample_groups():
    records = []
    for group in ["A", "B", "C", "D"]:
        for pot, team in enumerate([f"Team_{group}{i+1}" for i in range(4)]):
            records.append({"group": group, "team": team, "pot": pot + 1})
    return pd.DataFrame(records)