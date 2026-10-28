# -*- coding: utf-8 -*-
"""
01_download_data.py - downloads and prepares both datasets used in the project.

Project: "What makes a Steam game successful?"
Team:    Kyrylo Serhiienko (71897), Roman Zavoritnii (72086)

Dataset 1 (natural language) - Steam user reviews (UCSD, J. McAuley et al.)
    Hugging Face: recommender-system/steam-review-and-bundle-dataset
    file:         australian_user_reviews.json.gz
Dataset 2 (numerical data) - Steam store game metadata (FronkonGames)
    Hugging Face: FronkonGames/steam-games-dataset
    file:         data/train-00000-of-00001.parquet

Output (directory data/processed/):
    reviews.csv.gz  - one review per row, Steam account names replaced with numbers
    games.csv.gz    - one game per row, selected numerical and categorical columns
    (gzip-compressed CSV - pandas reads it without extra libraries: pd.read_csv('...csv.gz'))
    sources.json    - where the data came from (repository, file, commit, SHA-256, date, row count)

Usage (from the repository root):
    pip install -r requirements.txt
    python src/01_download_data.py

Both files are downloaded at the pinned commits below, so the results can be reproduced exactly.
If downloading from Hugging Face does not work, download both files manually in a browser
(links in README.md) and pass their paths:
    python src/01_download_data.py --reviews PATH/australian_user_reviews.json.gz --games PATH/train-00000-of-00001.parquet
"""
import argparse
import ast
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

SOURCE_REVIEWS = {
    "repo_id": "recommender-system/steam-review-and-bundle-dataset",
    "filename": "australian_user_reviews.json.gz",
    "revision": "8e9c6b569122fb20668f5da3b0f3970a853fc5d8",
}
SOURCE_GAMES = {
    "repo_id": "FronkonGames/steam-games-dataset",
    "filename": "data/train-00000-of-00001.parquet",
    "revision": "6a31fcb42082345acc817e176b28849b7c05f248",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_from_hf(source):
    """Downloads a file from a Hugging Face dataset repository at the pinned commit; returns (path, commit)."""
    from huggingface_hub import hf_hub_download

    directory = RAW_DIR / source["repo_id"].replace("/", "__")
    path = hf_hub_download(
        repo_id=source["repo_id"], filename=source["filename"], repo_type="dataset",
        revision=source["revision"], local_dir=str(directory),
    )
    return Path(path), source["revision"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset 1 - reviews
# ---------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def parse_date(text):
    """'Posted November 5, 2011.' -> (year, month, day); the year can be missing ('Posted July 15.')."""
    if not isinstance(text, str):
        return (np.nan, np.nan, np.nan)
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?", text)
    if not m or m.group(1).lower() not in MONTHS:
        return (np.nan, np.nan, np.nan)
    year = int(m.group(3)) if m.group(3) else np.nan
    return (year, MONTHS[m.group(1).lower()], int(m.group(2)))


def parse_helpful(text):
    """'2 of 3 people (67%) found this review helpful' -> (2, 3); 'No ratings yet' -> (0, 0)."""
    if not isinstance(text, str) or not text.strip():
        return (0, 0)
    m = re.search(r"([\d,]+)\s+of\s+([\d,]+)", text)
    if m:
        return (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")))
    m = re.search(r"([\d,]+)\s+(?:person|people)", text)
    if m:  # e.g. '1 person found this review helpful'
        n = int(m.group(1).replace(",", ""))
        return (n, n)
    return (0, 0)


def parse_funny(text):
    """'3 people found this review funny' -> 3; '' -> 0."""
    if not isinstance(text, str):
        return 0
    m = re.search(r"([\d,]+)\s+(?:person|people)", text)
    return int(m.group(1).replace(",", "")) if m else 0


def load_reviews(path):
    """Each line of the file is a dictionary in Python syntax (single quotes), not strict JSON."""
    rows, bad_lines, users = [], 0, 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except Exception:
                try:
                    record = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
            users += 1
            for r in record.get("reviews", []) or []:
                rows.append({
                    "user_id": record.get("user_id"),
                    "item_id": r.get("item_id"),
                    "posted": r.get("posted"),
                    "last_edited": r.get("last_edited"),
                    "funny": r.get("funny"),
                    "helpful": r.get("helpful"),
                    "recommend": r.get("recommend"),
                    "review": r.get("review"),
                })
    df = pd.DataFrame(rows)
    print(f"   users in file: {users}, reviews: {len(df)}, lines with syntax errors: {bad_lines}")

    # anonymisation: Steam account name -> consecutive number
    df["user_idx"] = pd.factorize(df["user_id"])[0]
    df = df.drop(columns=["user_id"])

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
    df["recommend"] = df["recommend"].map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else str(v).lower() == "true")
    df["review"] = df["review"].fillna("").astype(str)

    dates = df["posted"].map(parse_date)
    df["posted_year"] = pd.array([d[0] for d in dates], dtype="Float64").astype("Int64")
    df["posted_month"] = pd.array([d[1] for d in dates], dtype="Float64").astype("Int64")
    df["posted_day"] = pd.array([d[2] for d in dates], dtype="Float64").astype("Int64")
    df["edited"] = df["last_edited"].fillna("").astype(str).str.strip().ne("")

    helpful = df["helpful"].map(parse_helpful)
    df["helpful_yes"] = [h[0] for h in helpful]
    df["helpful_total"] = [h[1] for h in helpful]
    df["funny_count"] = df["funny"].map(parse_funny)

    columns = ["user_idx", "item_id", "recommend", "review", "posted_year", "posted_month", "posted_day",
               "edited", "helpful_yes", "helpful_total", "funny_count", "posted", "helpful", "funny"]
    return df[columns]


# ---------------------------------------------------------------------------
# Dataset 2 - games
# ---------------------------------------------------------------------------
def norm(name):
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


CANDIDATES = {  # target name -> possible column names in the file (after normalisation)
    "appid": ["appid", "app_id", "steam_appid", "id"],
    "name": ["name", "title"],
    "release_date": ["release_date"],
    "estimated_owners": ["estimated_owners"],
    "peak_ccu": ["peak_ccu"],
    "required_age": ["required_age"],
    "price": ["price"],
    "discount": ["discount"],
    "dlc_count": ["dlc_count", "discountdlc_count"],
    "windows": ["windows"],
    "mac": ["mac"],
    "linux": ["linux"],
    "metacritic_score": ["metacritic_score"],
    "user_score": ["user_score"],
    "positive": ["positive"],
    "negative": ["negative"],
    "achievements": ["achievements"],
    "recommendations": ["recommendations"],
    "average_playtime_forever": ["average_playtime_forever"],
    "average_playtime_2weeks": ["average_playtime_2weeks", "average_playtime_two_weeks"],
    "median_playtime_forever": ["median_playtime_forever"],
    "median_playtime_2weeks": ["median_playtime_2weeks", "median_playtime_two_weeks"],
    "supported_languages": ["supported_languages"],
    "full_audio_languages": ["full_audio_languages"],
    "developers": ["developers"],
    "publishers": ["publishers"],
    "categories": ["categories"],
    "genres": ["genres"],
    "tags": ["tags"],
}
NUMERIC = ["peak_ccu", "required_age", "price", "discount", "dlc_count", "metacritic_score", "user_score",
           "positive", "negative", "achievements", "recommendations", "average_playtime_forever",
           "average_playtime_2weeks", "median_playtime_forever", "median_playtime_2weeks"]
LIST_COLUMNS = ["supported_languages", "full_audio_languages", "developers", "publishers", "categories", "genres", "tags"]


def to_list(v):
    """List / dict / text '[...]' / 'a, b' -> list of strings (a tag dictionary is sorted by votes, highest first)."""
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    if isinstance(v, dict):
        pairs = [(k, w) for k, w in v.items() if w is not None and not (isinstance(w, float) and np.isnan(w))]
        try:
            pairs.sort(key=lambda kv: -float(kv[1]))
        except Exception:
            pass
        return [str(k) for k, _ in pairs]
    if isinstance(v, (list, tuple, np.ndarray)):
        result = []
        for x in v:
            if isinstance(x, dict):  # e.g. a list of structs {'tag': ..., 'count': ...}
                result.append(str(next(iter(x.values()))))
            elif x is not None:
                result.append(str(x))
        return result
    s = str(v).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return to_list(ast.literal_eval(s))
        except Exception:
            s = s.strip("[]{}")
    return [t.strip().strip("'\"") for t in s.split(",") if t.strip().strip("'\"")]


def load_games(path):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)
    return prepare_games(df)


def prepare_games(df):
    print(f"   rows: {len(df)}, columns in file: {len(df.columns)}")
    # the game id is sometimes stored as the index
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    lookup = {norm(c): c for c in df.columns}
    selected, missing = {}, []
    for target, candidates in CANDIDATES.items():
        for c in candidates:
            if c in lookup:
                selected[target] = lookup[c]
                break
        else:
            missing.append(target)
    print("   columns in file:", [str(c) for c in df.columns])
    if missing:
        print("   WARNING - columns not found:", missing)
    if "appid" not in selected or "positive" not in selected:
        sys.exit("Key columns are missing (appid / positive). Check the column list printed above.")

    games = pd.DataFrame({target: df[source] for target, source in selected.items()})
    games["appid"] = pd.to_numeric(games["appid"], errors="coerce").astype("Int64")
    for c in NUMERIC:
        if c in games:
            games[c] = pd.to_numeric(games[c], errors="coerce")
    for c in ["windows", "mac", "linux"]:
        if c in games:
            games[c] = games[c].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    if "release_date" in games:
        text = games["release_date"].astype(str)
        date = pd.to_datetime(text, errors="coerce", format="mixed")
        games["release_date"] = date
        games["release_year"] = date.dt.year.astype("Int64")
    if "estimated_owners" in games:
        numbers = games["estimated_owners"].astype(str).str.replace(",", "").str.extract(r"(\d+)\s*-\s*(\d+)")
        games["owners_low"] = pd.to_numeric(numbers[0], errors="coerce")
        games["owners_high"] = pd.to_numeric(numbers[1], errors="coerce")
    for c in LIST_COLUMNS:
        if c in games:
            lists = games[c].map(to_list)
            games[c] = lists.map(lambda l: ";".join(l))
            games[f"n_{c}"] = lists.map(len)

    before = len(games)
    games = games.dropna(subset=["appid"]).drop_duplicates(subset="appid")
    print(f"   after removing missing/duplicate appid: {len(games)} (removed {before - len(games)})")
    return games.reset_index(drop=True)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download and prepare the data for the Steam project.")
    parser.add_argument("--reviews", help="path to a manually downloaded australian_user_reviews.json.gz")
    parser.add_argument("--games", help="path to a manually downloaded parquet file with games")
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    sources = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": {}}

    for name, source, manual_path in [("reviews", SOURCE_REVIEWS, args.reviews),
                                      ("games", SOURCE_GAMES, args.games)]:
        print(f"\n### Dataset '{name}': {source['repo_id']} / {source['filename']}")
        if manual_path:
            path, commit = Path(manual_path), None
            print(f"   file given manually: {path}")
        else:
            path, commit = download_from_hf(source)
            print(f"   downloaded: {path} (commit {commit})")
        print(f"   file size: {path.stat().st_size / 1e6:.1f} MB")

        df = load_reviews(path) if name == "reviews" else load_games(path)
        output = PROC_DIR / f"{name}.csv.gz"
        df.to_csv(output, index=False, compression="gzip")
        print(f"   saved: {output.relative_to(ROOT).as_posix()} - {len(df)} rows, {df.shape[1]} columns, "
              f"{output.stat().st_size / 1e6:.1f} MB")
        sources["datasets"][name] = {
            "hf_repository": f"https://huggingface.co/datasets/{source['repo_id']}",
            "file": source["filename"],
            "commit": commit,
            "file_sha256": file_sha256(path),
            "file_size_mb": round(path.stat().st_size / 1e6, 2),
            "rows_after_preparation": int(len(df)),
            "columns_after_preparation": [str(c) for c in df.columns],
        }

    with open(PROC_DIR / "sources.json", "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)
    print("\nDone. Files in data/processed/: reviews.csv.gz, games.csv.gz, sources.json")


if __name__ == "__main__":
    main()
