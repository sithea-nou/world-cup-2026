import glob
import os
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import KAGGLE_MATCHES_DATASET, KAGGLE_RANKINGS_DATASET, RAW_DIR
from src.helpers import logger, ensure_dirs


def _find_and_extract_zip(dataset_name: str, out_dir: Path, expected_substring: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_candidates = list(RAW_DIR.glob(f"*{expected_substring}*.zip"))

    if not zip_candidates:
        all_zips = list(RAW_DIR.glob("*.zip"))
        zip_candidates = [z for z in all_zips if not any(
            z.name.startswith(prefix) for prefix in ["fifa_rankings", "international_matches"]
        )]

    if zip_candidates:
        zip_path = zip_candidates[0]
        logger.info(f"Extracting {zip_path.name} to {out_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        try:
            zip_path.unlink()
        except OSError:
            pass
        return True

    return False


def download_match_results(force: bool = False) -> Path:
    out_dir = RAW_DIR / "international_matches"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_files = ["results.csv", "shootouts.csv", "former_names.csv"]
    if not force and all((out_dir / f).exists() for f in expected_files):
        logger.info(f"Match results already downloaded at {out_dir}")
        return out_dir

    logger.info(f"Downloading dataset: {KAGGLE_MATCHES_DATASET}")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(KAGGLE_MATCHES_DATASET, path=str(out_dir), unzip=True)

    if all((out_dir / f).exists() for f in expected_files):
        logger.info(f"Match results downloaded to {out_dir}")
        for f in expected_files:
            if not (out_dir / f).exists():
                logger.warning(f"Expected file not found: {out_dir / f}")
        return out_dir

    nested_csvs = list(out_dir.rglob("*.csv"))
    if nested_csvs:
        for csv_file in nested_csvs:
            if csv_file.parent != out_dir:
                target = out_dir / csv_file.name
                if not target.exists():
                    import shutil
                    shutil.move(str(csv_file), str(target))
                    logger.info(f"Moved {csv_file.name} to {out_dir}")

    for subdir in out_dir.iterdir():
        if subdir.is_dir() and subdir.name != "__pycache__":
            try:
                import shutil
                shutil.rmtree(subdir)
            except OSError:
                pass

    if _find_and_extract_zip(KAGGLE_MATCHES_DATASET, out_dir, "international"):
        pass

    logger.info(f"Match results downloaded to {out_dir}")

    for f in expected_files:
        if not (out_dir / f).exists():
            logger.warning(f"Expected file not found: {out_dir / f}")

    return out_dir


def download_fifa_rankings(force: bool = False) -> Path:
    out_dir = RAW_DIR / "fifa_rankings"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_file = "fifa_ranking.csv"
    if not force and (out_dir / expected_file).exists():
        logger.info(f"FIFA rankings already downloaded at {out_dir}")
        return out_dir

    logger.info(f"Downloading dataset: {KAGGLE_RANKINGS_DATASET}")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(KAGGLE_RANKINGS_DATASET, path=str(out_dir), unzip=True)

    actual_csvs = list(out_dir.glob("*.csv"))
    if actual_csvs and not (out_dir / expected_file).exists():
        actual_csvs[0].rename(out_dir / expected_file)
        logger.info(f"Renamed {actual_csvs[0].name} to {expected_file}")

    nested_csvs = list(out_dir.rglob("*.csv"))
    if nested_csvs:
        for csv_file in nested_csvs:
            if csv_file.parent != out_dir:
                target = out_dir / csv_file.name
                if not target.exists():
                    import shutil
                    shutil.move(str(csv_file), str(target))

    if _find_and_extract_zip(KAGGLE_RANKINGS_DATASET, out_dir, "fifa"):
        actual_csvs = list(out_dir.glob("*.csv"))
        if actual_csvs and not (out_dir / expected_file).exists():
            actual_csvs[0].rename(out_dir / expected_file)
            logger.info(f"Renamed {actual_csvs[0].name} to {expected_file}")

    logger.info(f"FIFA rankings downloaded to {out_dir}")

    return out_dir


if __name__ == "__main__":
    ensure_dirs()
    download_match_results()
    download_fifa_rankings()