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


df = pd.read_csv(PROCESSED_DIR / "group_stage_probabilities.csv")

groups = sorted(df["group"].unique())
n_groups = len(groups)
n_cols = 4
n_rows = (n_groups + n_cols - 1) // n_cols

BG_COLOR = "#0a1628"
CARD_COLOR = "#121f36"
TEXT_COLOR = "#e8edf3"
ADVANCE_YES = "#1aff8c"
ADVANCE_NO = "#4a5568"
BAR_BG = "#1a2744"
SUBTLE_TEXT = "#7a8ba8"
ACCENT_GREEN = "#1aff8c"

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 6), facecolor=BG_COLOR)
if n_rows == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, group_letter in enumerate(groups):
    ax = axes[idx]
    ax.set_facecolor(CARD_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    group_data = df[df["group"] == group_letter].sort_values("prob_advance", ascending=True).reset_index(drop=True)

    teams = group_data["team"].tolist()
    y_pos = np.arange(len(teams))
    bar_height = 0.55

    ax.barh(y_pos, [100] * len(teams), height=bar_height, color=BAR_BG, zorder=1)

    advance_colors = [ADVANCE_YES if p >= 0.5 else ADVANCE_NO for p in group_data["prob_advance"]]
    bars = ax.barh(y_pos, group_data["prob_advance"] * 100, height=bar_height, color=advance_colors, alpha=0.85, zorder=2)

    for i, (bar, row) in enumerate(zip(bars, group_data.itertuples())):
        pct = row.prob_advance * 100
        if pct >= 10:
            ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}%", va="center", ha="right", fontsize=11, fontweight="bold",
                    color=BG_COLOR if pct > 50 else TEXT_COLOR)
        else:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}%", va="center", ha="left", fontsize=10, fontweight="bold",
                    color=advance_colors[i])

    for i, row in enumerate(group_data.itertuples()):
        team_label = row.team
        if len(team_label) > 14:
            team_label = team_label[:13] + "."

        flag_img = download_flag(row.team, size=120)
        if flag_img is not None:
            ax.text(-1.5, y_pos[i] + 0.15, team_label, va="center", ha="right", fontsize=11, fontweight="bold", color=TEXT_COLOR)
            imagebox = OffsetImage(flag_img, zoom=0.15)
            ab = AnnotationBbox(imagebox, (-4.5, y_pos[i] - 0.15), frameon=False, zorder=5)
            ax.add_artist(ab)
        else:
            ax.text(-2, y_pos[i], team_label, va="center", ha="right", fontsize=11, fontweight="bold", color=TEXT_COLOR)

    ax.set_yticks([])
    ax.set_xlim(-6, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=SUBTLE_TEXT)
    ax.tick_params(axis="x", length=0)
    ax.set_title(f"GROUP {group_letter}", fontsize=14, fontweight="bold", color=TEXT_COLOR,
                pad=10, loc="left")

    top2 = group_data.sort_values("prob_advance", ascending=False).head(2)
    if len(top2) >= 2:
        fav1 = top2.iloc[0]["team"]
        fav2 = top2.iloc[1]["team"]
        ax.text(105, y_pos[-1] + 0.8, f"Projected: {fav1}, {fav2}",
               va="bottom", ha="right", fontsize=8, color=ACCENT_GREEN, style="italic")

for idx in range(n_groups, len(axes)):
    axes[idx].set_facecolor(BG_COLOR)
    axes[idx].set_visible(False)

advance_patch = mpatches.Patch(color=ADVANCE_YES, label="Likely to advance (>50%)")
no_advance_patch = mpatches.Patch(color=ADVANCE_NO, label="Unlikely to advance")
legend = fig.legend(handles=[advance_patch, no_advance_patch], loc="lower center",
                   ncol=2, fontsize=12, frameon=False,
                   labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, 0.01))

fig.suptitle("WORLD CUP 2026 — GROUP STAGE PREDICTIONS",
            fontsize=22, fontweight="bold", color=TEXT_COLOR, y=0.98)

plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.94])

out_path = PROCESSED_DIR / "evaluation" / "group_stage_social.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Saved: {out_path}")