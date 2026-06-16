import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import urllib.request
import tempfile

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

FLAG_CACHE_DIR = Path(tempfile.gettempdir()) / "wc2026_flags"
FLAG_CACHE_DIR.mkdir(exist_ok=True)

COUNTRY_CODE_MAP = {
    "Mexico": "mx", "South Korea": "kr", "Czech Republic": "cz", "South Africa": "za",
    "Switzerland": "ch", "Canada": "ca", "Bosnia-Herzegovina": "ba", "Qatar": "qa",
    "Brazil": "br", "Scotland": "gb-sct", "Morocco": "ma", "Haiti": "ht",
    "United States": "us", "Australia": "au", "Turkey": "tr", "Paraguay": "py",
    "Germany": "de", "Ivory Coast": "ci", "Ecuador": "ec", "Curaçao": "cw",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Iran": "ir", "Egypt": "eg", "New Zealand": "nz",
    "Spain": "es", "Uruguay": "uy", "Cape Verde": "cv", "Saudi Arabia": "sa",
    "France": "fr", "Senegal": "sn", "Norway": "no", "Iraq": "iq",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "Colombia": "co", "Congo DR": "cd", "Uzbekistan": "uz",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}


def download_flag(country: str, size: int = 80) -> Image.Image | None:
    code = COUNTRY_CODE_MAP.get(country)
    if code is None:
        return None
    cache_path = FLAG_CACHE_DIR / f"{code}.png"
    if not cache_path.exists():
        url = f"https://flagcdn.com/w80/{code}.png"
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception:
            return None
    try:
        img = Image.open(cache_path).convert("RGBA").resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


validation_path = PROCESSED_DIR / "live_validation_report.csv"
if validation_path.exists():
    matches_df = pd.read_csv(validation_path)
else:
    import joblib
    from src.models.live_validation import validate_against_live
    model = joblib.load(PROCESSED_DIR / "models" / "best_model.joblib")
    feature_cols = joblib.load(PROCESSED_DIR / "models" / "feature_columns.joblib")
    result = validate_against_live(model, feature_cols)
    matches_df = result["results"]

n_correct = int(matches_df["correct"].sum())
n_total = len(matches_df)
accuracy = n_correct / n_total if n_total > 0 else 0

BG_COLOR = "#0a1628"
CARD_COLOR = "#121f36"
TEXT_COLOR = "#e8edf3"
SUBTLE_TEXT = "#7a8ba8"
CORRECT_COLOR = "#1aff8c"
WRONG_COLOR = "#ff5252"
PROB_HOME_COLOR = "#1aff8c"
PROB_DRAW_COLOR = "#4fc3f7"
PROB_AWAY_COLOR = "#ffab40"
SCORE_COLOR = "#4fc3f7"
BAR_BG = "#1a2744"

n_matches = len(matches_df)
fig, ax = plt.subplots(figsize=(18, max(9, n_matches * 1.0)), facecolor=BG_COLOR)
ax.set_facecolor(CARD_COLOR)
for spine in ax.spines.values():
    spine.set_visible(False)

y_pos = np.arange(n_matches)
bar_height = 0.6
bar_width = 30

for i in range(n_matches):
    y = y_pos[i]

    ax.barh(y, bar_width, height=bar_height, color=BAR_BG, zorder=1)

    row = matches_df.iloc[i]
    hw = row["prob_home_win"]
    dr = row["prob_draw"]
    aw = row["prob_away_win"]

    ax.barh(y, hw * bar_width, height=bar_height, left=0, color=PROB_HOME_COLOR, alpha=0.8, zorder=2)
    ax.barh(y, dr * bar_width, height=bar_height, left=hw * bar_width, color=PROB_DRAW_COLOR, alpha=0.8, zorder=2)
    ax.barh(y, aw * bar_width, height=bar_height, left=(hw + dr) * bar_width, color=PROB_AWAY_COLOR, alpha=0.8, zorder=2)

    if hw >= 0.15:
        ax.text(hw * bar_width / 2, y, f"{hw:.0%}", va="center", ha="center", fontsize=8, fontweight="bold", color=BG_COLOR)
    if dr >= 0.15:
        ax.text((hw + dr / 2) * bar_width, y, f"{dr:.0%}", va="center", ha="center", fontsize=8, fontweight="bold", color=BG_COLOR)
    if aw >= 0.15:
        ax.text((hw + dr + aw / 2) * bar_width, y, f"{aw:.0%}", va="center", ha="center", fontsize=8, fontweight="bold", color=BG_COLOR)

    home_short = row["home_team"][:12] if len(row["home_team"]) > 12 else row["home_team"]
    away_short = row["away_team"][:12] if len(row["away_team"]) > 12 else row["away_team"]

    home_flag = download_flag(row["home_team"], size=120)
    if home_flag is not None:
        imagebox = OffsetImage(home_flag, zoom=0.11)
        ab = AnnotationBbox(imagebox, (-6.5, y - 0.13), frameon=False, zorder=5)
        ax.add_artist(ab)
        ax.text(-4, y - 0.13, home_short, va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_COLOR)
    else:
        ax.text(-6, y, home_short, va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_COLOR)

    away_flag = download_flag(row["away_team"], size=120)
    if away_flag is not None:
        imagebox2 = OffsetImage(away_flag, zoom=0.11)
        ab2 = AnnotationBbox(imagebox2, (-6.5, y + 0.13), frameon=False, zorder=5)
        ax.add_artist(ab2)
        ax.text(-4, y + 0.13, away_short, va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_COLOR)
    else:
        ax.text(-6, y, away_short, va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_COLOR)

    score_str = f"{int(row['home_score'])} - {int(row['away_score'])}"
    ax.text(bar_width + 1.5, y + 0.12, score_str, va="center", ha="left", fontsize=10, fontweight="bold", color=SCORE_COLOR)
    ax.text(bar_width + 1.5, y - 0.12, row["actual_outcome"].replace("_", " ").upper(), va="center", ha="left", fontsize=7, color=SUBTLE_TEXT)

    icon = "✓" if row["correct"] else "✗"
    icon_color = CORRECT_COLOR if row["correct"] else WRONG_COLOR
    ax.text(bar_width + 8, y, icon, va="center", ha="center", fontsize=14, fontweight="bold", color=icon_color)

    pred_label = row["predicted_outcome"].replace("_", " ").upper()
    pred_color = CORRECT_COLOR if row["correct"] else WRONG_COLOR
    ax.text(bar_width + 10, y, f"Predicted: {pred_label}", va="center", ha="left", fontsize=8, color=pred_color)

ax.set_yticks([])
ax.set_xlim(-8, bar_width + 18)
ax.set_xticks([])
ax.invert_yaxis()

legend_patches = [
    mpatches.Patch(color=PROB_HOME_COLOR, alpha=0.8, label="Home Win Prob"),
    mpatches.Patch(color=PROB_DRAW_COLOR, alpha=0.8, label="Draw Prob"),
    mpatches.Patch(color=PROB_AWAY_COLOR, alpha=0.8, label="Away Win Prob"),
    mpatches.Patch(facecolor="none", edgecolor=CORRECT_COLOR, label=f"✓ Correct ({n_correct})"),
    mpatches.Patch(facecolor="none", edgecolor=WRONG_COLOR, label=f"✗ Wrong ({n_total - n_correct})"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9, frameon=True,
         facecolor=CARD_COLOR, edgecolor=SUBTLE_TEXT, labelcolor=TEXT_COLOR,
         bbox_to_anchor=(1.0, 1.0))

fig.suptitle(
    f"WORLD CUP 2026 — PREDICTION ACCURACY: {n_correct}/{n_total} ({accuracy:.0%})",
    fontsize=18, fontweight="bold", color=TEXT_COLOR, y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = PROCESSED_DIR / "evaluation" / "prediction_accuracy_social.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Saved: {out_path}")
print(f"Accuracy: {n_correct}/{n_total} ({accuracy:.0%})")