# -*- coding: utf-8 -*-
"""
04_eda_joined.py - joining both datasets: reviews (dataset 1) + game metadata (dataset 2).

Key:     item_id (reviews) = appid (games)
Input:   data/processed/reviews.csv.gz, data/processed/games.csv.gz
Output:  reports/stage1/figures/c*.png, reports/stage1/results/eda_joined.json
Usage:   python src/04_eda_joined.py
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency

from common import (BLUE, ORANGE, GREY, TEXT, save_figure, save_results, load_reviews, load_games,
                    wilson_interval, clean_reviews)

MIN_REVIEWS = 20   # minimum number of reviews of a game in dataset 1
MIN_RATINGS = 50   # minimum number of Steam ratings of a game (dataset 2)
PRICE_WORDS = re.compile(r"\b(?:price|prices|priced|money|worth|cheap|expensive|sale|refund|overpriced|dollars?|bucks|\$\d+)",
                         re.IGNORECASE)


def share_with_ci(bool_series):
    k, n = int(bool_series.sum()), int(bool_series.size)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "share": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi}


def main():
    print("=" * 70)
    print("EDA - joined datasets (reviews + games)")
    print("=" * 70)
    r = clean_reviews(load_reviews())   # same cleaning as in 02_eda_reviews.py
    g = load_games()
    g["n_ratings"] = g["positive"].fillna(0) + g["negative"].fillna(0)
    g["positive_share"] = g["positive"] / g["n_ratings"].replace(0, np.nan)
    g["free"] = g["price"].fillna(-1).eq(0)
    g["main_genre"] = g["genres"].str.split(";").str[0].replace("", np.nan)
    R = {"min_reviews": MIN_REVIEWS, "min_ratings": MIN_RATINGS}

    # ---------------- coverage
    games_in_reviews = r["item_id"].dropna().unique()
    found = np.isin(games_in_reviews, g["appid"].values)
    R["games_in_reviews"] = int(len(games_in_reviews))
    R["games_found_in_dataset2"] = int(found.sum())
    R["game_coverage"] = float(found.mean())
    rg = r.merge(g[["appid", "name", "price", "free", "main_genre", "genres", "release_year",
                    "n_ratings", "positive_share"]], left_on="item_id", right_on="appid", how="left")
    R["review_coverage"] = float(rg["appid"].notna().mean())
    R["games_missing_in_dataset2"] = int(len(games_in_reviews) - found.sum())
    R["reviews_of_games_missing_in_dataset2"] = int(rg["appid"].isna().sum())
    print(f"Games from reviews found in the games dataset: {R['games_found_in_dataset2']} of {R['games_in_reviews']} "
          f"({R['game_coverage']:.1%}); reviews with a matched game: {R['review_coverage']:.1%}")
    joined = rg[rg["appid"].notna()].copy()
    joined["free"] = joined["free"].astype(bool)

    # ---------------- agreement: % 'recommended' in dataset 1 vs % positive on Steam (dataset 2)
    per_game = joined.groupby("item_id").agg(n=("recommend", "size"), recommended=("recommend", "mean"),
                                             steam=("positive_share", "first"), n_ratings=("n_ratings", "first"),
                                             name=("name", "first"))
    comp = per_game[(per_game["n"] >= MIN_REVIEWS) & (per_game["n_ratings"] >= MIN_RATINGS)].copy()
    rho, p = spearmanr(comp["recommended"], comp["steam"]) if len(comp) >= 3 else (np.nan, np.nan)
    R["agreement"] = {"games": int(len(comp)), "spearman_rho": rho, "p": p,
                      "mean_difference_pp": float((comp["recommended"] - comp["steam"]).mean() * 100),
                      "median_abs_difference_pp": float((comp["recommended"] - comp["steam"]).abs().median() * 100)}
    print(f"Agreement (games with >= {MIN_REVIEWS} reviews and >= {MIN_RATINGS} ratings, n = {len(comp)}): "
          f"rho = {rho:.3f}, p = {p:.2e}, mean difference = {R['agreement']['mean_difference_pp']:+.1f} pp")
    comp["difference"] = comp["recommended"] - comp["steam"]
    R["largest_disagreements"] = comp.reindex(comp["difference"].abs().sort_values(ascending=False).index).head(8)[
        ["name", "n", "recommended", "steam"]].reset_index().to_dict(orient="records")

    # ---------------- % 'recommended' vs game features (review level)
    R["recommended_free"] = share_with_ci(joined.loc[joined["free"], "recommend"])
    R["recommended_paid"] = share_with_ci(joined.loc[~joined["free"], "recommend"])
    tab = pd.crosstab(joined["free"], joined["recommend"])
    chi2, p, dof, _ = chi2_contingency(tab)
    R["chi2_free_vs_recommended"] = {"chi2": chi2, "p": p, "df": dof}
    print(f"% 'recommended': free {R['recommended_free']['share']:.1%}, paid {R['recommended_paid']['share']:.1%}, "
          f"chi2 = {chi2:.1f}, p = {p:.2e}")

    top_genres = joined["main_genre"].value_counts()
    top_genres = top_genres[top_genres >= 200].head(8).index.tolist()
    R["recommended_by_genre"] = {x: share_with_ci(joined.loc[joined["main_genre"] == x, "recommend"]) for x in top_genres}
    if len(top_genres) >= 2:
        sub = joined[joined["main_genre"].isin(top_genres)]
        chi2, p, dof, _ = chi2_contingency(pd.crosstab(sub["main_genre"], sub["recommend"]))
        R["chi2_genre_vs_recommended"] = {"chi2": chi2, "p": p, "df": dof}

    # ---------------- mentions of price in the text
    joined["mentions_price"] = joined["review"].str.contains(PRICE_WORDS)
    R["mentions_price_share"] = float(joined["mentions_price"].mean())
    R["recommended_when_mentions_price"] = share_with_ci(joined.loc[joined["mentions_price"], "recommend"])
    R["recommended_when_no_price_mention"] = share_with_ci(joined.loc[~joined["mentions_price"], "recommend"])
    R["mentions_price_free"] = float(joined.loc[joined["free"], "mentions_price"].mean())
    R["mentions_price_paid"] = float(joined.loc[~joined["free"], "mentions_price"].mean())
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(joined["mentions_price"], joined["recommend"]))
    R["chi2_price_in_text_vs_recommended"] = {"chi2": chi2, "p": p, "df": dof}
    print(f"Reviews mentioning price: {R['mentions_price_share']:.1%}; % 'recommended' when mentioned "
          f"{R['recommended_when_mentions_price']['share']:.1%} vs not mentioned {R['recommended_when_no_price_mention']['share']:.1%}")

    # ================= FIGURES =================
    # c1: agreement of ratings in both datasets
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    size = 12 + 60 * np.sqrt(comp["n"] / comp["n"].max())
    ax.scatter(comp["steam"] * 100, comp["recommended"] * 100, s=size, color=BLUE, alpha=0.55,
               edgecolor="white", linewidth=0.5)
    ax.plot([0, 100], [0, 100], ls=":", color=GREY, lw=1.5, label="perfect agreement")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_xlabel("% positive ratings on Steam (dataset 2)")
    ax.set_ylabel("% 'recommended' reviews (dataset 1)")
    ax.set_title(f"The same game in both datasets (n = {len(comp)} games, rho = {rho:.2f})")
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, "c1_dataset_agreement")

    # c2: % 'recommended' by genre and by price mention
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    if top_genres:
        data = sorted(R["recommended_by_genre"].items(), key=lambda kv: kv[1]["share"])
        y = np.arange(len(data))
        sh = np.array([d["share"] for _, d in data]) * 100
        lo = np.array([d["ci_lo"] for _, d in data]) * 100
        hi = np.array([d["ci_hi"] for _, d in data]) * 100
        ax.errorbar(sh, y, xerr=[sh - lo, hi - sh], fmt="o", color=BLUE, ecolor=GREY, capsize=3, ms=6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{k} (n={d['n']})" for k, d in data])
        ax.set_xlabel("% 'recommended' reviews (95% Wilson interval)")
        ax.set_title("Reviews by the game's main genre")
        ax.grid(axis="y", visible=False)
    ax = axes[1]
    labels = ["mentions price", "no price mention"]
    vals = [R["recommended_when_mentions_price"], R["recommended_when_no_price_mention"]]
    sh = np.array([v["share"] for v in vals]) * 100
    err = np.array([[v["share"] - v["ci_lo"] for v in vals], [v["ci_hi"] - v["share"] for v in vals]]) * 100
    ax.bar(labels, sh, yerr=err, color=[ORANGE, BLUE], width=0.55, capsize=4, ecolor=GREY)
    for i, (v, d) in enumerate(zip(sh, vals)):
        ax.text(i, min(v + 6, 96), f"{v:.1f}%  (n={d['n']})", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% 'recommended' reviews")
    ax.set_title("Reviews that talk about price")
    save_figure(fig, "c2_recommended_by_features")

    save_results(R, "eda_joined")


if __name__ == "__main__":
    main()
