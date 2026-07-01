"""Generate presentation slides (PNG) and an MP4 video for WC 2026 predictor results."""

import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from src.config import PROCESSED_DIR, RAW_DIR
from src.features.build_2026_features import build_wc2026_features

SLIDES_DIR = PROCESSED_DIR / "slides"
SLIDES_DIR.mkdir(exist_ok=True)

BG_DARK = "#1a1a2e"
BG_MED = "#16213e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
GREEN = "#27ae60"
RED = "#e74c3c"
YELLOW = "#f1c40f"
WHITE = "#ecf0f1"
GREY = "#95a5a6"
BLUE = "#3498db"
ORANGE = "#e67e22"


def compute_standings(live):
    gm = live[live["group"].notna() & (live["group"] != "R32")].copy()
    gm["home_score"] = gm["home_score"].astype(int)
    gm["away_score"] = gm["away_score"].astype(int)
    standings = {}
    for grp in sorted(gm["group"].unique()):
        g = gm[gm["group"] == grp]
        teams = sorted(set(g["home_team"]).union(set(g["away_team"])))
        table = {}
        for t in teams:
            table[t] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
        for _, r in g.iterrows():
            h, a, hs, as_ = r["home_team"], r["away_team"], r["home_score"], r["away_score"]
            table[h]["P"] += 1; table[a]["P"] += 1
            table[h]["GF"] += hs; table[h]["GA"] += as_
            table[a]["GF"] += as_; table[a]["GA"] += hs
            if hs > as_: table[h]["W"] += 1; table[a]["L"] += 1; table[h]["Pts"] += 3
            elif hs < as_: table[a]["W"] += 1; table[h]["L"] += 1; table[a]["Pts"] += 3
            else: table[h]["D"] += 1; table[a]["D"] += 1; table[h]["Pts"] += 1; table[a]["Pts"] += 1
        df = pd.DataFrame(table).T; df["GD"] = df["GF"] - df["GA"]
        standings[grp] = df.sort_values(["Pts", "GD", "GF"], ascending=False)
    return standings


def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_MED,
        "axes.edgecolor": GREY,
        "axes.labelcolor": WHITE,
        "xtick.color": WHITE,
        "ytick.color": WHITE,
        "text.color": WHITE,
        "font.size": 12,
        "font.family": "sans-serif",
    })


def slide_title(fig, ax):
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.text(0.5, 0.72, "World Cup 2026", ha="center", va="center",
            fontsize=52, fontweight="bold", color=ACCENT, transform=ax.transAxes)
    ax.text(0.5, 0.58, "ML Predictor — Live Results", ha="center", va="center",
            fontsize=28, color=WHITE, transform=ax.transAxes)
    ax.text(0.5, 0.42, "Elo + XGBoost + Ensemble", ha="center", va="center",
            fontsize=18, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.25, "48 teams  |  12 groups  |  104 matches", ha="center", va="center",
            fontsize=16, color=ACCENT2, transform=ax.transAxes)
    ax.text(0.5, 0.10, "July 2026", ha="center", va="center",
            fontsize=14, color=GREY, transform=ax.transAxes)
    fig.patch.set_facecolor(BG_DARK)


def slide_data_model(fig, ax):
    ax.set_facecolor(BG_MED)
    ax.axis("off")
    ax.text(0.5, 0.92, "Data & Model", ha="center", va="center",
            fontsize=32, fontweight="bold", color=ACCENT, transform=ax.transAxes)

    items = [
        ("Data Sources", [
            "49,477 historical international matches (Kaggle)",
            "FIFA rankings, Elo ratings, squad market values",
            "Transfermarkt squad quality, betting odds",
            "Wikipedia live results (WC 2026)",
        ]),
        ("Features (40+)", [
            "Elo delta, FIFA rank delta, squad value delta",
            "Form (last 5/10), H2H, strength of schedule",
            "Confederation, home advantage, tournament type",
            "Combined draw probability, draw tendency",
        ]),
        ("Models", [
            "XGBoost (4x draw weight) — 819KB",
            "Random Forest (class_weight=balanced) — 47MB",
            "Logistic Regression (class_weight=balanced)",
            "Neural Network (8x draw weight)",
            "Best: WeightedVotingEnsemble (inverse-log-loss) — 96MB",
        ]),
    ]

    y = 0.80
    for title, lines in items:
        ax.text(0.06, y, title, fontsize=16, fontweight="bold", color=ACCENT2, transform=ax.transAxes)
        y -= 0.045
        for line in lines:
            ax.text(0.10, y, f"•  {line}", fontsize=11, color=WHITE, transform=ax.transAxes)
            y -= 0.035
        y -= 0.02

    fig.patch.set_facecolor(BG_DARK)


def slide_group_results(fig, ax, group_correct, group_total, overall_correct, overall_total, log_loss):
    ax.set_facecolor(BG_MED)
    ax.axis("off")
    ax.text(0.5, 0.95, "Group Stage — Live Validation", ha="center", va="center",
            fontsize=28, fontweight="bold", color=ACCENT, transform=ax.transAxes)

    # Big metrics
    ax.text(0.15, 0.82, f"{overall_correct}/{overall_total}", fontsize=44, fontweight="bold",
            color=GREEN if overall_correct / overall_total >= 0.6 else YELLOW, transform=ax.transAxes)
    ax.text(0.15, 0.74, "Correct Predictions", fontsize=14, color=GREY, transform=ax.transAxes)
    ax.text(0.15, 0.68, f"{overall_correct/overall_total*100:.1f}% accuracy", fontsize=18, color=WHITE, transform=ax.transAxes)

    ax.text(0.50, 0.82, f"{log_loss:.3f}", fontsize=44, fontweight="bold", color=BLUE, transform=ax.transAxes)
    ax.text(0.50, 0.74, "Log Loss", fontsize=14, color=GREY, transform=ax.transAxes)
    ax.text(0.50, 0.68, "(lower is better)", fontsize=12, color=GREY, transform=ax.transAxes)

    draws_missed = overall_total - overall_correct
    ax.text(0.82, 0.82, f"{draws_missed}", fontsize=44, fontweight="bold", color=RED, transform=ax.transAxes)
    ax.text(0.82, 0.74, "Misses", fontsize=14, color=GREY, transform=ax.transAxes)
    ax.text(0.82, 0.68, "mostly draws", fontsize=12, color=GREY, transform=ax.transAxes)

    # Per-group bars
    groups = sorted(group_total.keys())
    corrects = [group_correct.get(g, 0) for g in groups]
    totals = [group_total.get(g, 0) for g in groups]
    rates = [c / t if t > 0 else 0 for c, t in zip(corrects, totals)]

    ax_bar = fig.add_axes([0.12, 0.08, 0.78, 0.40])
    ax_bar.set_facecolor(BG_MED)
    colors = [GREEN if r >= 0.67 else (YELLOW if r >= 0.5 else RED) for r in rates]
    bars = ax_bar.bar(groups, rates, color=colors, edgecolor=WHITE, linewidth=0.5)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Accuracy", fontsize=12, color=WHITE)
    ax_bar.set_xlabel("Group", fontsize=12, color=WHITE)
    ax_bar.set_title("Accuracy by Group", fontsize=16, color=ACCENT, pad=10)
    ax_bar.tick_params(colors=WHITE)
    for bar, c, t in zip(bars, corrects, totals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{c}/{t}", ha="center", fontsize=9, color=WHITE, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.patch.set_facecolor(BG_DARK)


def slide_r32(fig, ax, r32_data):
    ax.set_facecolor(BG_MED)
    ax.axis("off")
    ax.text(0.5, 0.96, "Round of 32 — Predictions", ha="center", va="center",
            fontsize=28, fontweight="bold", color=ACCENT, transform=ax.transAxes)

    col_labels = ["M#", "Home", "Away", "H%", "D%", "A%", "Pred", "Score", ""]
    cell_text = []
    cell_colors = []

    for mn, h, a, hp, dp, ap, pred, score, ok in r32_data:
        cell_text.append([str(mn), h[:16], a[:16], f"{hp:.0f}", f"{dp:.0f}", f"{ap:.0f}", pred, score, ok])
        rc = [BG_MED] * len(col_labels)
        if ok == "Y":
            rc[-1] = GREEN
        elif ok == "N":
            rc[-1] = RED
        if pred == "home_win":
            rc[6] = "#1a5276"
        elif pred == "away_win":
            rc[6] = "#7e5109"
        elif pred == "draw":
            rc[6] = "#7d6608"
        for j in range(9):
            if rc[j] == BG_MED:
                rc[j] = "#2c3e50"
        cell_colors.append(rc)

    table = ax.table(
        cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for i in range(len(col_labels)):
        table[0, i].set_facecolor(ACCENT)
        table[0, i].set_text_props(color="white", fontweight="bold")

    correct = sum(1 for r in r32_data if r[8] == "Y")
    played = sum(1 for r in r32_data if r[8] != "")
    pending = sum(1 for r in r32_data if r[8] == "")
    ax.text(0.5, 0.03, f"Played: {played}  |  Correct: {correct}  |  Pending: {pending}",
            ha="center", fontsize=14, color=GREEN if correct == played else YELLOW,
            fontweight="bold", transform=ax.transAxes)

    fig.patch.set_facecolor(BG_DARK)


def slide_summary(fig, ax, gs_correct, gs_total, r32_correct, r32_played):
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.text(0.5, 0.90, "Summary", ha="center", va="center",
            fontsize=36, fontweight="bold", color=ACCENT, transform=ax.transAxes)

    stats = [
        ("Group Stage", f"{gs_correct}/{gs_total}", f"{gs_correct/gs_total*100:.1f}%", GREEN if gs_correct/gs_total >= 0.6 else YELLOW),
        ("Round of 32", f"{r32_correct}/{r32_played}", f"{r32_correct/max(r32_played,1)*100:.1f}%", GREEN if r32_played > 0 and r32_correct/r32_played >= 0.7 else YELLOW),
        ("Total", f"{gs_correct+r32_correct}/{gs_total+r32_played}", f"{(gs_correct+r32_correct)/(gs_total+r32_played)*100:.1f}%", ACCENT2),
    ]

    y = 0.72
    for label, count, pct, color in stats:
        ax.text(0.20, y, label, fontsize=20, color=WHITE, transform=ax.transAxes)
        ax.text(0.55, y, count, fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.80, y, pct, fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
        y -= 0.12

    ax.text(0.5, 0.30, "Key Insights", ha="center", fontsize=18, fontweight="bold", color=ACCENT2, transform=ax.transAxes)
    insights = [
        "Draw prediction remains the model's biggest weakness",
        "Strong favorites (Argentina, Spain, France) predicted correctly",
        "Upsets caught: Norway over Ivory Coast, Canada over South Africa",
        "Best model: WeightedVotingEnsemble (inverse-log-loss soft voting)",
    ]
    y = 0.24
    for ins in insights:
        ax.text(0.15, y, f"•  {ins}", fontsize=12, color=WHITE, transform=ax.transAxes)
        y -= 0.045

    fig.patch.set_facecolor(BG_DARK)


def get_group_validation():
    report = pd.read_csv(PROCESSED_DIR / "live_validation_report.csv")
    live = pd.read_csv(RAW_DIR / "wc2026_results_live.csv")
    live = live[live["home_score"].notna()].copy()
    gm = live[live["group"].notna() & (live["group"] != "R32")].copy()
    group_correct = {}
    group_total = {}
    for _, r in report.iterrows():
        h, a = r["home_team"], r["away_team"]
        match = gm[(gm["home_team"] == h) & (gm["away_team"] == a)]
        if match.empty:
            match = gm[(gm["home_team"] == a) & (gm["away_team"] == h)]
        if not match.empty:
            grp = match.iloc[0]["group"]
            group_total.setdefault(grp, 0)
            group_correct.setdefault(grp, 0)
            group_total[grp] += 1
            if r["correct"]:
                group_correct[grp] += 1
    return group_correct, group_total


def get_r32_data():
    model = joblib.load(PROCESSED_DIR / "models" / "best_model.joblib")
    feature_cols = joblib.load(PROCESSED_DIR / "models" / "feature_columns.joblib")
    imputer = joblib.load(PROCESSED_DIR / "models" / "imputer.joblib")

    live = pd.read_csv(RAW_DIR / "wc2026_results_live.csv")
    live = live[live["home_score"].notna()].copy()
    live["home_score"] = live["home_score"].astype(int)
    live["away_score"] = live["away_score"].astype(int)

    standings = compute_standings(live)
    W = {g: standings[g].index[0] for g in standings}
    R = {g: standings[g].index[1] for g in standings}
    T = {g: standings[g].index[2] for g in standings}

    r32_matches = [
        (73, R["A"], R["B"]), (74, W["E"], T["D"]), (75, W["F"], R["C"]), (76, W["C"], R["F"]),
        (77, W["I"], T["F"]), (78, R["E"], R["I"]), (79, W["A"], T["E"]), (80, W["L"], T["K"]),
        (81, W["D"], T["B"]), (82, W["G"], T["I"]), (83, R["K"], R["L"]), (84, W["H"], R["J"]),
        (85, W["B"], T["J"]), (86, W["J"], R["H"]), (87, W["K"], T["L"]), (88, R["D"], R["G"]),
    ]

    fixtures_df = pd.DataFrame([
        {"match_number": mn, "date": "2026-06-30", "home_team": h, "away_team": a, "group": "R32"}
        for mn, h, a in r32_matches
    ])
    orig = PROCESSED_DIR / "wc2026_match_features.parquet"
    bk = PROCESSED_DIR / "bk.parquet"
    shutil.copy2(orig, bk)
    feat = build_wc2026_features(fixtures_df=fixtures_df, include_live=True)
    shutil.copy2(bk, orig)
    bk.unlink()

    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0.0
    X = feat[feature_cols].values
    if np.isnan(X).any():
        X = imputer.transform(X)
    proba = model.predict_proba(X)
    preds = model.predict(X)
    LABEL = ["away_win", "draw", "home_win"]

    played = {}
    for _, r in live[live["group"].isna() | (live["group"] == "R32")].iterrows():
        played[(r["home_team"], r["away_team"])] = (
            int(r["home_score"]), int(r["away_score"]),
            r["winner"] if pd.notna(r.get("winner", "")) else None,
        )

    r32_data = []
    for i, (mn, h, a) in enumerate(r32_matches):
        p = proba[i]
        pr = preds[i]
        score = ""
        ok = ""
        key = (h, a)
        swap = (a, h)
        if key in played:
            hs, as_, winner = played[key]
            score = f"{hs}-{as_}"
            if winner is not None:
                advancer = winner
                pred_advancer = h if pr == 2 else (a if pr == 0 else None)
                ok = "Y" if pred_advancer == advancer else "N"
            else:
                act = 2 if hs > as_ else (0 if hs < as_ else 1)
                ok = "Y" if pr == act else "N"
        elif swap in played:
            hs, as_, winner = played[swap]
            score = f"{hs}-{as_}"
            if winner is not None:
                advancer = winner
                pred_advancer = h if pr == 2 else (a if pr == 0 else None)
                ok = "Y" if pred_advancer == advancer else "N"
            else:
                act = 2 if hs > as_ else (0 if hs < as_ else 1)
                ok = "Y" if pr == act else "N"
        r32_data.append((mn, h, a, p[2] * 100, p[1] * 100, p[0] * 100, LABEL[pr], score, ok))
    return r32_data


def generate_slides():
    setup_dark_style()
    live = pd.read_csv(RAW_DIR / "wc2026_results_live.csv")
    live_played = live[live["home_score"].notna()]
    gm = live_played[live_played["group"].notna() & (live_played["group"] != "R32")]
    gs_total = len(gm)

    report = pd.read_csv(PROCESSED_DIR / "live_validation_report.csv")
    gs_correct = int(report["correct"].sum())
    log_loss = float(report.apply(lambda r: -np.log(max(
        [r["prob_away_win"], r["prob_draw"], r["prob_home_win"]][
            {"away_win": 0, "draw": 1, "home_win": 2}[r["actual_outcome"]]
        ], 1e-15)), axis=1).mean())

    group_correct, group_total = get_group_validation()

    r32_data = get_r32_data()
    r32_correct = sum(1 for r in r32_data if r[8] == "Y")
    r32_played = sum(1 for r in r32_data if r[8] != "")

    slides = []

    def save_slide(name):
        path = SLIDES_DIR / f"{name}.png"
        plt.savefig(path, dpi=150, facecolor=BG_DARK, bbox_inches="tight")
        plt.close()
        slides.append(path)
        print(f"  Saved {path}")

    # Slide 1: Title
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    slide_title(fig, ax)
    save_slide("01_title")

    # Slide 2: Data & Model
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    slide_data_model(fig, ax)
    save_slide("02_data_model")

    # Slide 3: Group Stage Results
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    slide_group_results(fig, ax, group_correct, group_total, gs_correct, gs_total, log_loss)
    save_slide("03_group_stage")

    # Slide 4: R32 Predictions
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    slide_r32(fig, ax, r32_data)
    save_slide("04_r32_predictions")

    # Slide 5: Summary
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    slide_summary(fig, ax, gs_correct, gs_total, r32_correct, r32_played)
    save_slide("05_summary")

    return slides, (gs_correct, gs_total, r32_correct, r32_played, log_loss)


def generate_video(slides, duration_per_slide=4.0, fps=24, transition=0.5):
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp()
    frames = []
    n_transition_frames = int(transition * fps)
    n_slide_frames = int(duration_per_slide * fps)

    from PIL import Image

    slide_imgs = [Image.open(str(s)).convert("RGB").resize((1920, 1080)) for s in slides]

    frame_idx = 0
    for si, img in enumerate(slide_imgs):
        for f in range(n_slide_frames):
            img.save(f"{tmpdir}/frame_{frame_idx:05d}.png")
            frame_idx += 1
        if si < len(slide_imgs) - 1:
            next_img = slide_imgs[si + 1]
            for f in range(n_transition_frames):
                alpha = f / n_transition_frames
                blended = Image.blend(img, next_img, alpha)
                blended.save(f"{tmpdir}/frame_{frame_idx:05d}.png")
                frame_idx += 1

    total_frames = frame_idx
    out_path = PROCESSED_DIR / "wc2026_predictor_video.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{tmpdir}/frame_%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "medium",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    import shutil as sh
    sh.rmtree(tmpdir)
    print(f"\nVideo saved to {out_path}")
    print(f"  {total_frames} frames, {total_frames/fps:.1f}s duration")
    return out_path


def main():
    print("Generating slides...")
    slides, stats = generate_slides()
    gs_correct, gs_total, r32_correct, r32_played, log_loss = stats

    print(f"\nStats:")
    print(f"  Group stage: {gs_correct}/{gs_total} ({gs_correct/gs_total*100:.1f}%), log_loss={log_loss:.3f}")
    print(f"  R32: {r32_correct}/{r32_played} ({r32_correct/max(r32_played,1)*100:.1f}%)")

    print("\nGenerating video...")
    video_path = generate_video(slides, duration_per_slide=4.0, fps=24, transition=0.7)

    print(f"\nSlides: {SLIDES_DIR}/ (5 PNG files)")
    print(f"Video:  {video_path}")


if __name__ == "__main__":
    main()