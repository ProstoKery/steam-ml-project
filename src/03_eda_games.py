# -*- coding: utf-8 -*-
"""
03_eda_games.py - exploratory data analysis of dataset 2 (Steam game metadata, numerical data).

Input:   data/processed/games.csv.gz (created by src/01_download_data.py)
Output:  reports/stage1/figures/g*.png, reports/stage1/results/eda_games.json
Usage:   python src/03_eda_games.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr, chi2_contingency

from common import BLUE, ORANGE, GREY, TEXT, save_figure, save_results, load_games, mann_whitney

MIN_RATINGS = 50          # a game needs at least this many ratings for its positive share to be reliable
SUCCESS_THRESHOLD = 0.80  # >= 80% positive ratings (on Steam: "Very Positive")


def main():
    print("=" * 70)
    print("EDA - dataset 2: Steam game metadata")
    print("=" * 70)
    g = load_games()
    R = {"min_ratings": MIN_RATINGS, "success_threshold": SUCCESS_THRESHOLD}

    g["n_ratings"] = g["positive"].fillna(0) + g["negative"].fillna(0)
    g["positive_share"] = np.where(g["n_ratings"] > 0, g["positive"] / g["n_ratings"].replace(0, np.nan), np.nan)
    g["free"] = g["price"].fillna(-1).eq(0)
    if "owners_low" in g:
        g["owners_mid"] = (g["owners_low"] + g["owners_high"]) / 2
    g["main_genre"] = g["genres"].str.split(";").str[0].replace("", np.nan)

    # ---------------- structure and missing values
    R["n_games"] = len(g)
    R["n_columns"] = g.shape[1]
    R["duplicate_appid"] = int(g["appid"].duplicated().sum())
    numeric = [c for c in ["price", "peak_ccu", "required_age", "dlc_count", "metacritic_score", "user_score",
                           "positive", "negative", "achievements", "recommendations",
                           "average_playtime_forever", "median_playtime_forever", "n_supported_languages"]
               if c in g]
    R["missing_and_zeros"] = {c: {"missing_%": float(g[c].isna().mean() * 100), "zeros_%": float((g[c] == 0).mean() * 100)}
                              for c in numeric}
    R["numeric_summary"] = g[numeric].describe(percentiles=[.25, .5, .75, .99]).T.round(3).to_dict(orient="index")
    R["without_release_date"] = int(g["release_date"].isna().sum()) if "release_date" in g else None
    R["without_genre"] = int(g["genres"].eq("").sum())
    R["without_tags_share"] = float(g["tags"].eq("").mean()) if "tags" in g else None
    R["zero_owners_and_no_ratings"] = int(((g.get("owners_high", 1) == 0) & (g["positive"].fillna(0) + g["negative"].fillna(0) == 0)).sum())
    print(f"Games: {R['n_games']}, columns: {R['n_columns']}, duplicate appid: {R['duplicate_appid']}")
    for c, v in R["missing_and_zeros"].items():
        print(f"   {c:<26} missing {v['missing_%']:5.1f}%   zeros {v['zeros_%']:5.1f}%")

    # ---------------- distributions
    R["free_share"] = float(g["free"].mean())
    paid = g.loc[g["price"] > 0, "price"]
    R["paid_price_summary"] = paid.describe(percentiles=[.25, .5, .75, .9, .99]).to_dict()
    R["games_without_ratings"] = float((g["n_ratings"] == 0).mean())
    R["n_ratings_summary"] = g["n_ratings"].describe(percentiles=[.5, .75, .9, .99]).to_dict()
    rated = g[g["n_ratings"] >= MIN_RATINGS].copy()
    R["games_min_ratings"] = len(rated)
    R["games_min_ratings_share"] = len(rated) / len(g)
    R["positive_share_summary"] = rated["positive_share"].describe().to_dict()
    R["success_share"] = float((rated["positive_share"] >= SUCCESS_THRESHOLD).mean())
    years = g["release_year"].dropna().astype(int)
    R["games_per_year"] = years.value_counts().sort_index().to_dict()
    R["platforms"] = {p: float(g[p].astype(str).str.lower().eq("true").mean()) for p in ["windows", "mac", "linux"] if p in g}
    genres = g["genres"].str.split(";").explode()
    genres = genres[genres.ne("")]
    R["top15_genres"] = genres.value_counts().head(15).to_dict()
    if "estimated_owners" in g:
        R["owner_ranges"] = g["estimated_owners"].value_counts().head(12).to_dict()
    print(f"Free games: {R['free_share']:.1%}; games with >= {MIN_RATINGS} ratings: {len(rated)} "
          f"({R['games_min_ratings_share']:.1%}); without any rating: {R['games_without_ratings']:.1%}")
    print(f"Median positive share (>= {MIN_RATINGS} ratings): {rated['positive_share'].median():.3f}; "
          f"'success' (>= {SUCCESS_THRESHOLD:.0%}): {R['success_share']:.1%}")

    # ---------------- Spearman correlations (games with >= MIN_RATINGS ratings)
    corr_cols = {"positive_share": "% positive", "n_ratings": "number of ratings", "price": "price",
                 "peak_ccu": "peak players (CCU)", "owners_mid": "owners (estimate)",
                 "median_playtime_forever": "median playtime", "achievements": "achievements",
                 "dlc_count": "number of DLC", "n_supported_languages": "number of languages",
                 "required_age": "required age", "release_year": "release year", "n_tags": "number of tags"}
    corr_cols = {k: v for k, v in corr_cols.items() if k in rated and rated[k].nunique(dropna=True) > 1}   # no constant columns
    corr = rated[list(corr_cols)].corr(method="spearman")
    R["spearman_correlation_matrix"] = corr.round(3).rename(index=corr_cols, columns=corr_cols).to_dict()
    R["spearman_with_positive_share"] = corr["positive_share"].drop("positive_share").sort_values().to_dict()
    print("Spearman correlation with % positive:",
          {corr_cols[k]: round(v, 3) for k, v in R["spearman_with_positive_share"].items()})

    # ---------------- statistical tests
    T = {}
    T["free_vs_paid_positive_share"] = mann_whitney(rated.loc[rated["free"], "positive_share"],
                                                    rated.loc[~rated["free"], "positive_share"])
    top_genres = rated["main_genre"].value_counts()
    top_genres = top_genres[top_genres >= 100].head(8).index.tolist()
    groups = [rated.loc[rated["main_genre"] == x, "positive_share"].dropna() for x in top_genres]
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        T["kruskal_main_genre"] = {"H": h, "p": p, "genres": top_genres,
                                   "medians": {x: float(gr.median()) for x, gr in zip(top_genres, groups)},
                                   "counts": {x: int(len(gr)) for x, gr in zip(top_genres, groups)}}
    meta = rated[rated["metacritic_score"] > 0]
    if len(meta) > 10:
        r, p = spearmanr(meta["metacritic_score"], meta["positive_share"])
        T["spearman_metacritic_vs_players"] = {"rho": r, "p": p, "n": len(meta)}
    r, p = spearmanr(rated["price"], rated["positive_share"], nan_policy="omit")
    T["spearman_price_vs_positive_share"] = {"rho": r, "p": p, "n": int(rated["price"].notna().sum())}
    tab = pd.crosstab(rated["free"], rated["positive_share"] >= SUCCESS_THRESHOLD)
    chi2, p, dof, _ = chi2_contingency(tab)
    T["chi2_free_vs_success"] = {"chi2": chi2, "p": p, "df": dof,
                                 "success_free": float((rated.loc[rated["free"], "positive_share"] >= SUCCESS_THRESHOLD).mean()),
                                 "success_paid": float((rated.loc[~rated["free"], "positive_share"] >= SUCCESS_THRESHOLD).mean())}
    R["tests"] = T
    t = T["free_vs_paid_positive_share"]
    print(f"Free vs paid (% positive): medians {t['median_1']:.3f} vs {t['median_2']:.3f}, p = {t['p']:.2e}")
    if "kruskal_main_genre" in T:
        print(f"Kruskal-Wallis (main genre): H = {T['kruskal_main_genre']['H']:.1f}, p = {T['kruskal_main_genre']['p']:.2e}")
    if "spearman_metacritic_vs_players" in T:
        s = T["spearman_metacritic_vs_players"]
        print(f"Metacritic vs players: rho = {s['rho']:.3f} (n = {s['n']}), p = {s['p']:.2e}")

    # ================= FIGURES =================
    # g1: number of games released per year
    fig, ax = plt.subplots(figsize=(9, 3.6))
    per_year = years[(years >= 2006)].value_counts().sort_index()
    last = int(per_year.index.max())
    colors = [GREY if y_ == last else BLUE for y_ in per_year.index]
    ax.bar(per_year.index, per_year.values, color=colors, width=0.75)
    ax.text(last, per_year.loc[last], f"{last}:\npartial year", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(range(2006, last + 1, 2))
    ax.set_xlabel("release year")
    ax.set_ylabel("number of games")
    ax.set_title(f"Games released on Steam per year (before 2006: {int((years < 2006).sum())} games)")
    save_figure(fig, "g1_games_per_year")

    # g2: price / number of ratings / positive share
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ax = axes[0]
    ax.hist(paid.clip(upper=70), bins=np.arange(0, 72.5, 2.5), color=BLUE, edgecolor="white")
    ax.set_xlabel("price of paid games [USD, capped at 70]")
    ax.set_ylabel("number of games")
    ax.set_title(f"Price (free: {R['free_share']:.0%} of games)")
    ax = axes[1]
    positive_counts = g.loc[g["n_ratings"] > 0, "n_ratings"]
    ax.hist(np.log10(positive_counts), bins=40, color=BLUE, edgecolor="white")
    ax.axvline(np.log10(MIN_RATINGS), color=ORANGE, ls="--", lw=1.5, label=f"threshold {MIN_RATINGS} ratings")
    ax.set_xlabel("number of ratings (log10)")
    ax.set_title(f"Ratings per game ({R['games_without_ratings']:.0%} of games have none)")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.hist(rated["positive_share"] * 100, bins=np.arange(0, 102.5, 2.5), color=BLUE, edgecolor="white")
    ax.axvline(SUCCESS_THRESHOLD * 100, color=ORANGE, ls="--", lw=1.5, label=f"'success' threshold {SUCCESS_THRESHOLD:.0%}")
    ax.set_xlabel("% positive ratings")
    ax.set_title(f"Positive share (games with ≥{MIN_RATINGS} ratings)")
    ax.legend(fontsize=8, loc="upper left")
    save_figure(fig, "g2_distributions")

    # g3: genres - counts and distribution of % positive
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    gt = pd.Series(R["top15_genres"]).head(12)[::-1]
    ax.barh(gt.index, gt.values, color=BLUE, height=0.65)
    ax.set_xlabel("number of games (a game can have several genres)")
    ax.set_title("Most common genres")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    if len(groups) >= 2:
        order = np.argsort([np.median(x) for x in groups])
        ax.boxplot([groups[i] * 100 for i in order], orientation="horizontal", widths=0.6, showfliers=False,
                   tick_labels=[f"{top_genres[i]} (n={len(groups[i])})" for i in order],
                   patch_artist=True, boxprops=dict(facecolor="#cfe1f7", color=BLUE),
                   medianprops=dict(color=ORANGE, lw=2), whiskerprops=dict(color=GREY),
                   capprops=dict(color=GREY))
        ax.set_xlabel("% positive ratings")
        ax.set_title(f"% positive by main genre (≥{MIN_RATINGS} ratings)")
        ax.grid(axis="y", visible=False)
    save_figure(fig, "g3_genres")

    # g4: Spearman correlation matrix
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    labels = [corr_cols[c] for c in corr.columns]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman coefficient", shrink=0.8)
    ax.set_title(f"Spearman correlations (games with ≥{MIN_RATINGS} ratings, n = {len(rated)})")
    save_figure(fig, "g4_correlations")

    # g5: free vs paid and critics (Metacritic) vs players
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.boxplot([rated.loc[rated["free"], "positive_share"] * 100, rated.loc[~rated["free"], "positive_share"] * 100],
               tick_labels=[f"free (n={int(rated['free'].sum())})", f"paid (n={int((~rated['free']).sum())})"],
               widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#cfe1f7", color=BLUE),
               medianprops=dict(color=ORANGE, lw=2))
    ax.set_ylabel("% positive ratings")
    ax.set_title(f"Free vs paid (U test: p = {t['p']:.1e})")
    ax = axes[1]
    if len(meta) > 10:
        hb = ax.hexbin(meta["metacritic_score"], meta["positive_share"] * 100, gridsize=30, cmap="Blues", mincnt=1)
        fig.colorbar(hb, ax=ax, label="number of games")
        s = T["spearman_metacritic_vs_players"]
        ax.set_title(f"Critics vs players (rho = {s['rho']:.2f}, n = {s['n']})")
        ax.set_xlabel("Metacritic score (critics)")
        ax.set_ylabel("% positive ratings (players)")
    save_figure(fig, "g5_tests")

    save_results(R, "eda_games")


if __name__ == "__main__":
    main()
