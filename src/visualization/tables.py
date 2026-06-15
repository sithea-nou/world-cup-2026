import pandas as pd
from src.config import PROCESSED_DIR
from src.helpers import logger


def format_power_rankings(df: pd.DataFrame, top_n: int = 30) -> str:
    logger.info("Formatting power rankings...")

    df = df.sort_values("prob_winner", ascending=False).head(top_n).reset_index(drop=True)

    lines = []
    lines.append("=" * 100)
    lines.append(f"{'WORLD CUP 2026 POWER RANKINGS':^100}")
    lines.append("=" * 100)
    lines.append(
        f"{'Rank':<6}{'Team':<25}{'Win%':<10}{'Final%':<10}{'SF%':<10}{'QF%':<10}{'Ro16%':<10}"
    )
    lines.append("-" * 100)

    for i, row in df.iterrows():
        lines.append(
            f"{i + 1:<6}{row['team']:<25}"
            f"{row['prob_winner'] * 100:<10.2f}"
            f"{row['prob_final'] * 100:<10.2f}"
            f"{row['prob_sf'] * 100:<10.2f}"
            f"{row['prob_qf'] * 100:<10.2f}"
            f"{row['prob_ro16'] * 100:<10.2f}"
        )

    lines.append("=" * 100)

    result = "\n".join(lines)
    logger.info(f"\n{result}")

    return result


def format_group_tables(group_probs: pd.DataFrame) -> str:
    logger.info("Formatting group tables...")

    lines = []
    lines.append("=" * 80)
    lines.append(f"{'WORLD CUP 2026 GROUP STAGE PROBABILITIES':^80}")
    lines.append("=" * 80)

    for group in sorted(group_probs["group"].unique()):
        group_data = group_probs[group_probs["group"] == group].sort_values("prob_advance", ascending=False)

        lines.append(f"\nGroup {group}:")
        lines.append("-" * 80)
        lines.append(
            f"{'Team':<25}{'1st%':<10}{'2nd%':<10}{'3rd%':<10}{'4th%':<10}{'Advance%':<10}"
        )
        lines.append("-" * 80)

        for _, row in group_data.iterrows():
            lines.append(
                f"{row['team']:<25}"
                f"{row['prob_1st'] * 100:<10.2f}"
                f"{row['prob_2nd'] * 100:<10.2f}"
                f"{row['prob_3rd'] * 100:<10.2f}"
                f"{row['prob_4th'] * 100:<10.2f}"
                f"{row['prob_advance'] * 100:<10.2f}"
            )

    lines.append("=" * 80)

    result = "\n".join(lines)
    logger.info(f"\n{result}")

    return result


def format_bracket_summary(df: pd.DataFrame, top_n: int = 16) -> str:
    logger.info("Formatting bracket summary...")

    df = df.sort_values("prob_winner", ascending=False).reset_index(drop=True)

    lines = []
    lines.append("=" * 70)
    lines.append(f"{'WORLD CUP 2026 BRACKET SUMMARY':^70}")
    lines.append("=" * 70)
    lines.append(
        f"{'Team':<25}{'Winner':<10}{'Final':<10}{'SF':<10}{'QF':<10}{'R16':<10}"
    )
    lines.append("-" * 70)

    for _, row in df.head(top_n).iterrows():
        lines.append(
            f"{row['team']:<25}"
            f"{row['prob_winner'] * 100:<10.2f}"
            f"{row['prob_final'] * 100:<10.2f}"
            f"{row['prob_sf'] * 100:<10.2f}"
            f"{row['prob_qf'] * 100:<10.2f}"
            f"{row['prob_ro16'] * 100:<10.2f}"
        )

    lines.append("=" * 70)

    result = "\n".join(lines)
    logger.info(f"\n{result}")

    return result


def format_match_predictions(predictions: pd.DataFrame) -> str:
    logger.info("Formatting match predictions...")

    lines = []
    lines.append("=" * 90)
    lines.append(f"{'WORLD CUP 2026 MATCH PREDICTIONS':^90}")
    lines.append("=" * 90)

    groups = sorted(predictions["group"].unique()) if "group" in predictions.columns and predictions["group"].notna().any() else [""]
    for group in groups:
        if group:
            group_data = predictions[predictions["group"] == group].reset_index(drop=True)
        else:
            group_data = predictions.reset_index(drop=True)

        if group_data.empty:
            continue

        lines.append(f"\nGroup {group}:")
        lines.append("-" * 90)
        lines.append(
            f"{'Home':<22}{'Away':<22}{'H Win%':<10}{'Draw%':<10}{'A Win%':<10}{'Prediction':<10}"
        )
        lines.append("-" * 90)

        for _, row in group_data.iterrows():
            lines.append(
                f"{row['home_team']:<22}"
                f"{row['away_team']:<22}"
                f"{row['prob_home_win'] * 100:<10.1f}"
                f"{row['prob_draw'] * 100:<10.1f}"
                f"{row['prob_away_win'] * 100:<10.1f}"
                f"{row['prediction']:<10}"
            )

    lines.append("=" * 90)

    result = "\n".join(lines)
    logger.info(f"\n{result}")

    return result