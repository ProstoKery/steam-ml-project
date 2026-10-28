# -*- coding: utf-8 -*-
"""
02_eda_reviews.py - exploratory data analysis of dataset 1 (Steam user reviews, text).

Input:   data/processed/reviews.csv.gz (created by src/01_download_data.py)
Output:  reports/stage1/figures/r*.png, reports/stage1/results/eda_reviews.json
Usage:   python src/02_eda_reviews.py
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from common import (COLOR_RECOMMENDED, COLOR_NOT_RECOMMENDED, GREY, TEXT, BLUE,
                    save_figure, save_results, load_reviews, mann_whitney, wilson_interval, clean_reviews)

TOKEN = re.compile(r"[a-z][a-z]+(?:'[a-z]+)?")   # words of at least 2 letters
STOP = set(ENGLISH_STOP_WORDS) | {"s", "t", "don't", "it's", "i'm", "can't", "doesn't", "didn't", "isn't",
                                   "you're", "i've", "that's", "there's", "won't", "wasn't", "aren't", "i'd", "ll"}
NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "nor", "cannot", "don't", "doesn't", "didn't",
             "isn't", "can't", "won't", "wasn't", "aren't"}
DOMAIN_WORDS = {"game", "games", "play", "played", "playing", "plays"}   # appear everywhere, say nothing about the rating
EXCLUDED_FROM_COMPARISON = (STOP - NEGATIONS) | DOMAIN_WORDS
COLLECTION_YEAR = 2016   # Steam omits the year for posts from the current year; the data has no post dated 2016


def tokens(text):
    return TOKEN.findall(text.lower())


def language(text, toks):
    """Simple language heuristic: Cyrillic / CJK characters / share of English function words."""
    if re.search(r"[Ѐ-ӿ]", text):
        return "Cyrillic (e.g. Russian)"
    if re.search(r"[぀-ヿ一-鿿가-힯]", text):
        return "CJK characters"
    if len(toks) < 5:
        return "too short to tell"
    share = sum(t in ENGLISH_STOP_WORDS for t in toks) / len(toks)
    return "English" if share >= 0.15 else "other / unknown"


def log_odds(counts_a, counts_b, alpha0=500.0, min_count=20):
    """Log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008) -> z-score for each word."""
    total = counts_a + counts_b
    words = [w for w, c in total.items() if c >= min_count]
    n_a, n_b, n = sum(counts_a.values()), sum(counts_b.values()), sum(total.values())
    results = []
    for w in words:
        a_w = alpha0 * total[w] / n
        ya, yb = counts_a[w], counts_b[w]
        delta = (np.log((ya + a_w) / (n_a + alpha0 - ya - a_w)) - np.log((yb + a_w) / (n_b + alpha0 - yb - a_w)))
        var = 1 / (ya + a_w) + 1 / (yb + a_w)
        results.append((w, delta / np.sqrt(var), ya, yb))
    return pd.DataFrame(results, columns=["word", "z", "n_recommended", "n_not_recommended"]).sort_values("z")


def main():
    print("=" * 70)
    print("EDA - dataset 1: Steam user reviews")
    print("=" * 70)
    df = load_reviews()
    R = {}

    # ---------------- basic information
    df["n_words"] = df["review"].str.split().str.len().fillna(0).astype(int)
    df["n_chars"] = df["review"].str.len()
    empty = df["review"].str.strip().eq("")
    R["n_reviews"] = len(df)
    R["n_users"] = df["user_idx"].nunique()
    R["n_games"] = df["item_id"].nunique()
    R["missing_per_column"] = df.isna().sum().to_dict()
    R["empty_reviews"] = int(empty.sum())
    R["duplicate_user_game_pairs"] = int(df.duplicated(subset=["user_idx", "item_id"]).sum())
    R["duplicate_identical_text"] = int(df.loc[~empty].duplicated(subset=["review"]).sum())
    R["reviews_per_user"] = df.groupby("user_idx").size().describe().to_dict()
    print(f"Reviews: {R['n_reviews']}, users: {R['n_users']}, games: {R['n_games']}")
    print(f"Empty reviews: {R['empty_reviews']}, duplicate (user, game) pairs: {R['duplicate_user_game_pairs']}, "
          f"repeated identical text: {R['duplicate_identical_text']}")
    dup = df[df.duplicated(subset=["user_idx", "item_id"], keep=False)]
    R["duplicate_pairs_same_text_share"] = float(dup.groupby(["user_idx", "item_id"])["review"].nunique().eq(1).mean())
    R["most_common_identical_texts"] = df.loc[~empty, "review"].str.strip().value_counts().head(10).to_dict()

    # ---------------- cleaning: empty reviews and repeated (user, game) pairs - an artefact of data collection
    df = clean_reviews(df)
    empty = df["review"].str.strip().eq("")
    R["after_cleaning_reviews"] = len(df)
    R["after_cleaning_users"] = df["user_idx"].nunique()
    R["after_cleaning_games"] = df["item_id"].nunique()
    print(f"After cleaning: {len(df)} reviews, {R['after_cleaning_users']} users, "
          f"{R['after_cleaning_games']} games")

    # ---------------- target variable
    k = int(df["recommend"].sum())
    R["recommended_count"] = k
    R["recommended_share"] = k / len(df)
    R["recommended_ci95"] = wilson_interval(k, len(df))
    print(f"Share of 'recommended' reviews: {R['recommended_share']:.3%} (95% CI {R['recommended_ci95'][0]:.3%}"
          f" - {R['recommended_ci95'][1]:.3%})")

    # ---------------- time
    R["reviews_without_year"] = int(df["posted_year"].isna().sum())
    R["assumed_collection_year"] = COLLECTION_YEAR
    years = df.copy()
    years["posted_year"] = years["posted_year"].fillna(COLLECTION_YEAR).astype(int)
    year_tab = years.groupby("posted_year").agg(n=("recommend", "size"), recommended=("recommend", "mean"))
    R["reviews_per_year"] = year_tab.reset_index().to_dict(orient="records")
    tab = pd.crosstab(years["posted_year"], years["recommend"])
    chi2, p, dof, _ = chi2_contingency(tab)
    R["chi2_year_vs_recommended"] = {"chi2": chi2, "p": p, "df": dof}
    print(f"Reviews without a year in the date: {R['reviews_without_year']} (assumed year {COLLECTION_YEAR}); "
          f"chi2 (year x recommended) = {chi2:.1f}, p = {p:.2e}")
    print(year_tab.round(3).to_string())

    # ---------------- text length
    non_empty = df.loc[~empty]
    R["length_words_summary"] = non_empty["n_words"].describe(percentiles=[.25, .5, .75, .9, .99]).to_dict()
    R["very_short_up_to_3_words"] = float((non_empty["n_words"] <= 3).mean())
    R["test_length_not_recommended_vs_recommended"] = mann_whitney(non_empty.loc[~non_empty["recommend"], "n_words"],
                                                                   non_empty.loc[non_empty["recommend"], "n_words"])
    t = R["test_length_not_recommended_vs_recommended"]
    print(f"Length (words): median not recommended = {t['median_1']:.0f}, recommended = {t['median_2']:.0f}, "
          f"U = {t['U']:.0f}, p = {t['p']:.2e}, effect r = {t['effect_rank_biserial']:.3f}")

    # ---------------- language (heuristic)
    toks = non_empty["review"].map(tokens)
    languages = [language(tx, tk) for tx, tk in zip(non_empty["review"], toks)]
    R["language_heuristic"] = pd.Series(languages).value_counts(normalize=True).to_dict()
    print("Language (heuristic):", {k_: f"{v:.1%}" for k_, v in R["language_heuristic"].items()})

    # ---------------- games in the review dataset
    per_game = df.groupby("item_id").agg(n=("recommend", "size"), recommended=("recommend", "mean"))
    R["reviews_per_game_summary"] = per_game["n"].describe(percentiles=[.5, .9, .99]).to_dict()
    R["games_with_1_review"] = int((per_game["n"] == 1).sum())
    ranking = per_game["n"].sort_values(ascending=False)
    R["top10_games_share_of_reviews"] = float(ranking.head(10).sum() / len(df))
    R["top15_games"] = ranking.head(15).reset_index().to_dict(orient="records")
    min30 = per_game[per_game["n"] >= 30]
    R["games_min30_reviews"] = len(min30)
    R["games_min30_recommended_share_summary"] = min30["recommended"].describe().to_dict()

    # ---------------- helpfulness / humour
    R["reviews_with_helpfulness_votes"] = float((df["helpful_total"] > 0).mean())
    R["reviews_marked_funny"] = float((df["funny_count"] > 0).mean())
    R["test_helpful_votes_not_recommended_vs_recommended"] = mann_whitney(df.loc[~df["recommend"], "helpful_total"],
                                                                          df.loc[df["recommend"], "helpful_total"])
    R["edited_share"] = float(df["edited"].astype(str).str.lower().eq("true").mean())

    # ---------------- vocabulary
    counts_pos, counts_neg = Counter(), Counter()
    for tk, rec in zip(toks, non_empty["recommend"]):
        (counts_pos if rec else counts_neg).update(tk)
    all_counts = counts_pos + counts_neg
    R["n_tokens"] = sum(all_counts.values())
    R["vocabulary_size"] = len(all_counts)
    R["words_occurring_once"] = sum(1 for c in all_counts.values() if c == 1)
    top = [(w, c) for w, c in all_counts.most_common(300) if w not in STOP][:20]
    R["most_common_words_without_stopwords"] = top
    lo = log_odds(counts_pos, counts_neg)
    lo = lo[~lo["word"].isin(EXCLUDED_FROM_COMPARISON)]
    R["distinctive_words_recommended"] = lo.tail(15).iloc[::-1][["word", "z"]].values.tolist()
    R["distinctive_words_not_recommended"] = lo.head(15)[["word", "z"]].values.tolist()
    print("Words typical of 'recommended':", [w for w, _ in R["distinctive_words_recommended"]])
    print("Words typical of 'not recommended':", [w for w, _ in R["distinctive_words_not_recommended"]])

    # ================= FIGURES =================
    # r1: class balance, reviews per year, share of 'recommended' per year (separate panels, one axis each)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.5, 1.5]})
    ax = axes[0]
    values = [k, len(df) - k]
    bars = ax.bar(["recommended", "not recommended"], values, color=[COLOR_RECOMMENDED, COLOR_NOT_RECOMMENDED], width=0.6)
    for b_, v in zip(bars, values):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:,}\n({v / len(df):.1%})".replace(",", " "),
                ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("number of reviews")
    ax.set_title("Does the player recommend the game?")
    yt = year_tab[year_tab["n"] >= 30]
    ax = axes[1]
    ax.bar([f"{y_}*" if y_ == COLLECTION_YEAR else str(y_) for y_ in yt.index.astype(int)], yt["n"],
           color=BLUE, width=0.65)
    ax.set_ylabel("number of reviews")
    ax.set_title("Reviews per year (* = no year in the date)")
    ax = axes[2]
    ci = [wilson_interval(int(round(p_ * n_)), int(n_)) for p_, n_ in zip(yt["recommended"], yt["n"])]
    lo = np.array([c[0] for c in ci]) * 100
    hi = np.array([c[1] for c in ci]) * 100
    x = [f"{y_}*" if y_ == COLLECTION_YEAR else str(y_) for y_ in yt.index.astype(int)]
    ax.fill_between(x, lo, hi, color="#cfe1f7", label="95% Wilson interval")
    ax.plot(x, yt["recommended"] * 100, "o-", color=BLUE, lw=2, label="% 'recommended'")
    ax.set_ylim(max(0, lo.min() - 10), 100)
    ax.set_ylabel("% 'recommended' reviews")
    ax.set_title("Share of 'recommended' per year")
    ax.legend(fontsize=8, loc="lower left")
    save_figure(fig, "r1_classes_and_years")

    # r2: review length by class - empirical CDF (share of reviews with at most x words)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for rec, color, label in [(True, COLOR_RECOMMENDED, "recommended"), (False, COLOR_NOT_RECOMMENDED, "not recommended")]:
        x = np.sort(non_empty.loc[non_empty["recommend"] == rec, "n_words"].values)
        y = np.arange(1, len(x) + 1) / len(x) * 100
        med = np.median(x)
        ax.step(x, y, where="post", color=color, lw=2, label=f"{label} (median {med:.0f} words)")
        ax.plot([med], [50], "o", color=color, ms=6)
    ax.axhline(50, color=GREY, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1, non_empty["n_words"].quantile(0.999))
    ax.set_ylim(0, 100)
    ax.set_xlabel("review length [number of words, log scale]")
    ax.set_ylabel("% of reviews with length ≤ x")
    ax.set_title("Review length by class (empirical CDF)")
    ax.legend(loc="lower right")
    save_figure(fig, "r2_review_length")

    # r3: concentration of reviews on games + share of 'recommended' per game
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ranked = per_game["n"].sort_values(ascending=False).values
    ax.loglog(np.arange(1, len(ranked) + 1), ranked, color=BLUE, lw=2)
    ax.set_xlabel("game rank by number of reviews (log)")
    ax.set_ylabel("number of reviews of the game (log)")
    ax.set_title(f"Long tail: 10 games = {R['top10_games_share_of_reviews']:.0%} of reviews")
    ax = axes[1]
    ax.hist(min30["recommended"] * 100, bins=np.arange(0, 102.5, 5), color=BLUE, edgecolor="white")
    ax.axvline(R["recommended_share"] * 100, color=COLOR_NOT_RECOMMENDED, ls="--", lw=1.5,
               label=f"whole dataset average ({R['recommended_share']:.0%})")
    ax.set_xlabel("% 'recommended' reviews of the game")
    ax.set_ylabel("number of games")
    ax.set_title(f"Games with ≥30 reviews (n = {len(min30)})")
    ax.legend(fontsize=8, loc="upper left")
    save_figure(fig, "r3_games_in_reviews")

    # r4: distinctive words
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, color, title in [
        (axes[0], R["distinctive_words_recommended"], COLOR_RECOMMENDED, "'recommended'"),
        (axes[1], R["distinctive_words_not_recommended"], COLOR_NOT_RECOMMENDED, "'not recommended'")]:
        words = [w for w, _ in data][::-1]
        z = [abs(v) for _, v in data][::-1]
        ax.barh(words, z, color=color, height=0.65)
        ax.set_xlabel("|z| log-odds (strength of association with the class)")
        ax.set_title(f"Words most typical of {title}")
        ax.grid(axis="y", visible=False)
    save_figure(fig, "r4_distinctive_words")

    save_results(R, "eda_reviews")


if __name__ == "__main__":
    main()
