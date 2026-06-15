import argparse
import sys
import time
from pathlib import Path

from src.config import DATA_DIR, RAW_DIR, PROCESSED_DIR
from src.helpers import logger, ensure_dirs, setup_kaggle_credentials


def run_scraping():
    logger.info("=" * 60)
    logger.info("STEP 1: Data Scraping & Download")
    logger.info("=" * 60)

    from src.scraping.download_kaggle import download_match_results, download_fifa_rankings
    from src.scraping.scrape_fifa_rankings import scrape_current_fifa_rankings, merge_rankings
    from src.scraping.scrape_world_cup_2026 import scrape_wc2026_groups, scrape_wc2026_fixtures
    from src.scraping.scrape_odds import fetch_outright_odds, scrape_match_odds
    from src.scraping.scrape_live_results import scrape_live_results
    from src.scraping.scrape_historical_world_cups import scrape_historical_brackets

    try:
        setup_kaggle_credentials()
    except SystemExit:
        logger.warning("Kaggle credentials not available. Some data may be missing.")

    download_match_results()
    download_fifa_rankings()

    current_rankings = scrape_current_fifa_rankings()
    hist_path = RAW_DIR / "fifa_rankings" / "fifa_ranking.csv"
    if hist_path.exists():
        merge_rankings(hist_path, current_rankings)

    scrape_wc2026_groups()
    scrape_wc2026_fixtures()

    fetch_outright_odds()
    scrape_match_odds()

    scrape_live_results()
    scrape_historical_brackets()

    logger.info("Scraping complete!")


def run_features(include_live: bool = False):
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 60)

    from src.features.elo import EloRatingSystem
    from src.features.build_features import build_match_features
    from src.features.build_2026_features import build_wc2026_features

    import pandas as pd

    elo_system = EloRatingSystem()

    results_path = RAW_DIR / "international_matches" / "results.csv"
    if not results_path.exists():
        logger.error("Match results not found. Run scraping step first.")
        return

    matches_df = pd.read_csv(results_path)
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    matches_df["home_team"] = matches_df["home_team"].apply(
        lambda x: x if pd.isna(x) else str(x)
    )
    matches_df["away_team"] = matches_df["away_team"].apply(
        lambda x: x if pd.isna(x) else str(x)
    )

    from src.helpers import normalize_team_name
    matches_df["home_team"] = matches_df["home_team"].apply(
        lambda x: normalize_team_name(str(x)) if not pd.isna(x) else x
    )
    matches_df["away_team"] = matches_df["away_team"].apply(
        lambda x: normalize_team_name(str(x)) if not pd.isna(x) else x
    )

    elo_system.compute_elo_ratings(matches_df)

    features_df = build_match_features(matches_df, elo_system, include_live=include_live)
    logger.info(f"Built {len(features_df)} match features")

    wc2026_df = build_wc2026_features(include_live=include_live)
    if not wc2026_df.empty:
        logger.info(f"Built {len(wc2026_df)} WC 2026 match features")

    logger.info("Feature engineering complete!")


def run_train(include_live: bool = False):
    logger.info("=" * 60)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 60)

    import pandas as pd
    from src.models.train import (
        split_data,
        train_xgboost,
        train_random_forest,
        train_logistic_regression,
        train_neural_net,
        save_all_models,
    )

    features_path = PROCESSED_DIR / "match_features.parquet"
    if not features_path.exists():
        logger.error("Features not found. Run features step first.")
        return

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["outcome"])

    if include_live:
        live_path = RAW_DIR / "wc2026_results_live.csv"
        if live_path.exists():
            live_df = pd.read_csv(live_path)
            if "date" in live_df.columns:
                live_df["date"] = pd.to_datetime(live_df["date"])
            logger.info(f"Including {len(live_df)} live results in training data")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_cols, train_df, val_df, test_df) = split_data(df)

    logger.info("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train, X_val, y_val)

    logger.info("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    logger.info("Training Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)

    logger.info("Training Neural Network...")
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
            val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
            logger.info(f"{name}: train_acc={train_acc:.4f}, val_acc={val_acc_str}")

    logger.info("Model training complete!")


def run_ensemble():
    logger.info("=" * 60)
    logger.info("STEP 3b: Ensemble Building")
    logger.info("=" * 60)

    import joblib
    import pandas as pd
    from src.models.ensemble import build_best_ensemble
    from src.models.train import split_data

    models_dir = PROCESSED_DIR / "models"

    models = {}
    for model_file in models_dir.glob("*.joblib"):
        if model_file.stem in ("feature_columns", "best_model", "imputer"):
            continue
        name = model_file.stem.replace("_", " ").title()
        model = joblib.load(model_file)
        models[name] = {"model": model, "name": name}

    features_path = PROCESSED_DIR / "match_features.parquet"
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["outcome"])

    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_cols, train_df, val_df, test_df) = split_data(df)

    best = build_best_ensemble(models, X_train, y_train, X_val, y_val)
    logger.info(f"Best ensemble: {best['name']}")


def run_evaluate():
    logger.info("=" * 60)
    logger.info("STEP 4: Model Evaluation")
    logger.info("=" * 60)

    import joblib
    import pandas as pd
    from src.models.evaluate import evaluate_model, generate_evaluation_report
    from src.models.train import split_data

    features_path = PROCESSED_DIR / "match_features.parquet"
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["outcome"])

    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_cols, train_df, val_df, test_df) = split_data(df)

    models_dir = PROCESSED_DIR / "models"
    all_results = {}

    for model_file in models_dir.glob("*.joblib"):
        if model_file.stem in ("feature_columns", "best_model", "imputer"):
            continue

        name = model_file.stem.replace("_", " ").title()
        model = joblib.load(model_file)
        if not hasattr(model, "predict"):
            logger.warning(f"Skipping {name} - no predict method")
            continue
        results = evaluate_model(model, X_test, y_test, name)
        results["model"] = model
        all_results[name] = results

    # Evaluate best ensemble
    best_model_path = models_dir / "best_model.joblib"
    if best_model_path.exists():
        best_model = joblib.load(best_model_path)
        if hasattr(best_model, "predict"):
            results = evaluate_model(best_model, X_test, y_test, "BestEnsemble")
            results["model"] = best_model
            all_results["BestEnsemble"] = results

    if all_results:
        generate_evaluation_report(all_results, feature_cols)

    logger.info("Evaluation complete!")


def run_live_validate():
    logger.info("=" * 60)
    logger.info("STEP 4b: Live Validation")
    logger.info("=" * 60)

    import joblib
    from src.models.live_validation import validate_against_live

    models_dir = PROCESSED_DIR / "models"
    model = joblib.load(models_dir / "best_model.joblib")
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")

    validate_against_live(model, feature_cols)


def run_simulate(model_name: str = "xgboost", n_simulations: int = 1000):
    logger.info("=" * 60)
    logger.info("STEP 5: Tournament Simulation")
    logger.info("=" * 60)

    from src.simulation.simulator import run_simulation

    results = run_simulation(n_simulations=n_simulations, model_name=model_name)

    if not results.empty:
        logger.info("\nTop 10 teams by winning probability:")
        for i, row in results.head(10).iterrows():
            logger.info(f"  {i + 1}. {row['team']}: {row['prob_winner']:.4f}")

    logger.info("Simulation complete!")


def run_visualize():
    logger.info("=" * 60)
    logger.info("STEP 6: Visualization")
    logger.info("=" * 60)

    import pandas as pd
    from src.visualization.plots import (
        plot_tournament_probabilities,
        plot_round_probabilities,
    )
    from src.visualization.tables import format_power_rankings, format_bracket_summary

    results_path = PROCESSED_DIR / "tournament_probabilities.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)

        plot_tournament_probabilities(results, top_n=20)
        plot_round_probabilities(results)

        format_power_rankings(results)
        format_bracket_summary(results)
    else:
        logger.warning("Tournament probabilities not found. Run simulation first.")

    logger.info("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Predictor Pipeline")
    parser.add_argument("--step", choices=["scraping", "features", "train", "ensemble", "evaluate", "simulate", "visualize", "live-validate"], help="Run a specific step")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--retrain", action="store_true", help="Retrain models including live WC2026 data")
    parser.add_argument("--live-validate", action="store_true", help="Validate against live results")
    parser.add_argument("--n-simulations", type=int, default=None, help="Number of Monte Carlo simulations")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment (install deps, create dirs)")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")

    start_time = time.time()

    ensure_dirs()

    if args.setup_only:
        logger.info("Environment setup complete.")
        return

    if args.all:
        n_sims = args.n_simulations if args.n_simulations else 1000
        run_scraping()
        run_features(include_live=args.retrain)
        run_train(include_live=args.retrain)
        run_ensemble()
        run_evaluate()
        if args.live_validate:
            run_live_validate()
        run_simulate(model_name="xgboost", n_simulations=n_sims)
        run_visualize()
    elif args.step == "scraping":
        run_scraping()
    elif args.step == "features":
        run_features(include_live=args.retrain)
    elif args.step == "train":
        run_train(include_live=args.retrain)
    elif args.step == "ensemble":
        run_ensemble()
    elif args.step == "evaluate":
        run_evaluate()
    elif args.step == "simulate":
        n_sims = args.n_simulations if args.n_simulations else 1000
        run_simulate(model_name="xgboost", n_simulations=n_sims)
    elif args.step == "visualize":
        run_visualize()

    if args.live_validate and not args.all:
        run_live_validate()

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()