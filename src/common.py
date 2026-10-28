# -*- coding: utf-8 -*-
"""Shared settings for the analysis scripts: paths, chart colours, saving results, statistics helpers."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "stage1"
FIGURES = REPORT / "figures"
RESULTS = REPORT / "results"
for directory in (FIGURES, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)

# colours (categorical palette checked for colour-vision deficiency)
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PURPLE, RED, GREY, TEXT = "#6250d6", "#e34948", "#8a8984", "#52514e"
COLOR_RECOMMENDED, COLOR_NOT_RECOMMENDED = BLUE, ORANGE

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": "#b5b4ad",
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def save_figure(fig, name):
    path = FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   [figure] {path.relative_to(ROOT).as_posix()}")


def to_json(obj):
    """Converts numpy/pandas types to plain Python types (for JSON output)."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def save_results(results, name):
    path = RESULTS / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(results), f, ensure_ascii=False, indent=2)
    print(f"   [results] {path.relative_to(ROOT).as_posix()}")


def load_reviews():
    df = pd.read_csv(DATA / "reviews.csv.gz", low_memory=False)
    df["review"] = df["review"].fillna("").astype(str)
    df["recommend"] = df["recommend"].astype(str).str.lower().eq("true")
    return df


def clean_reviews(df):
    """Drops empty reviews and repeated (user, game) pairs - in the raw data these are identical copies."""
    df = df.loc[df["review"].str.strip().ne("")]
    return df.drop_duplicates(subset=["user_idx", "item_id"]).reset_index(drop=True)


def load_games():
    df = pd.read_csv(DATA / "games.csv.gz", low_memory=False)
    for c in ["supported_languages", "full_audio_languages", "developers", "publishers", "categories", "genres", "tags"]:
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    if "release_date" in df:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def mann_whitney(a, b):
    """Mann-Whitney U test + effect size (rank-biserial r = 1 - 2U/(n1*n2))."""
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(u), "p": float(p), "n1": int(len(a)), "n2": int(len(b)),
            "median_1": float(np.median(a)), "median_2": float(np.median(b)),
            "effect_rank_biserial": float(1 - 2 * u / (len(a) * len(b)))}


def wilson_interval(k, n, z=1.96):
    """95% Wilson confidence interval for the proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)
