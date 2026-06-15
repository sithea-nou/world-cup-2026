import json
import logging
import os
import stat
from pathlib import Path

from src.config import DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR

logger = logging.getLogger("worldcup")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory: {d}")


def setup_kaggle_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"

    if kaggle_file.exists():
        logger.info(f"Kaggle credentials found at {kaggle_file}")
        return

    logger.info("Kaggle credentials not found. Setting up...")
    logger.info("Go to https://www.kaggle.com/settings → API → Create New Token")
    logger.info("This downloads a kaggle.json file. Place it at ~/.kaggle/kaggle.json")
    logger.info("")
    logger.info("Alternatively, enter your credentials below:")

    username = input("Kaggle username: ").strip()
    key = input("Kaggle API key: ").strip()

    if not username or not key:
        logger.error("Kaggle credentials are required to download datasets.")
        raise SystemExit("Cannot continue without Kaggle credentials.")

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    creds = {"username": username, "key": key}
    kaggle_file.write_text(json.dumps(creds))
    kaggle_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    logger.info(f"Kaggle credentials saved to {kaggle_file}")


def normalize_team_name(name: str) -> str:
    from src.config import TEAM_NAME_MAPPING

    if not isinstance(name, str):
        return str(name)
    normalized = name.strip()
    return TEAM_NAME_MAPPING.get(normalized, normalized)


def load_cached_data(path: Path, force_refresh: bool = False):
    if not force_refresh and path.exists():
        import pandas as pd

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    return None