# What makes a Steam game successful?

Machine learning project for the **Machine Learning** course (Computer Science, Vistula University in Warsaw, winter semester 2026/2027).

**Team:** Kyrylo Serhiienko ([@ProstoKery](https://github.com/ProstoKery)), Roman Zavoritnii ([@yourromanbm-dot](https://github.com/yourromanbm-dot))

What makes players rate a Steam game well, and can it be predicted? We combine two public datasets from Hugging Face:
59k Steam user reviews labelled "recommended / not recommended" (UCSD, McAuley Lab) and metadata of 124k Steam games
(FronkonGames). The datasets are joined on the Steam game ID, so we can compare what players write with numbers that describe the game.

## Project status

| Part | Scope | Status |
| --- | --- | --- |
| Stage 1 | data, exploratory data analysis, research questions | done – [report](reports/stage1/report_stage1.pdf) |
| Next | supervised and unsupervised learning, NLP and transformer models, Gradio interface | planned |

## Data

| | Dataset 1: reviews (text) | Dataset 2: games (numbers) |
| --- | --- | --- |
| Source | [recommender-system/steam-review-and-bundle-dataset](https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset) – a copy of the dataset by J. McAuley's group (UCSD) | [FronkonGames/steam-games-dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) (DOI 10.57967/hf/0511) |
| File | `australian_user_reviews.json.gz` (commit `8e9c6b56`) | `data/train-00000-of-00001.parquet` (commit `6a31fcb4`) |
| Size | 59,305 reviews, 25,458 users, 3,682 games | 124,146 games |
| Period | reviews from 2010–2016 | games released 1997–2026 |
| License | not stated in the repository – we cite the authors' publications (below) | CC BY 4.0 (dataset card) |

The two datasets share the Steam game ID (`item_id` = `appid`).
Steam account names in the reviews were replaced with numbers.

Publications the authors of dataset 1 ask to cite:
- W.-C. Kang, J. McAuley. *Self-attentive sequential recommendation.* ICDM 2018.
- M. Wan, J. McAuley. *Item recommendation on monotonic behavior chains.* RecSys 2018.
- A. Pathak, K. Gupta, J. McAuley. *Generating and personalizing bundle recommendations on Steam.* SIGIR 2017.

## Repository structure

```
data/processed/           prepared data: reviews.csv.gz, games.csv.gz, sources.json (commit and SHA-256 of the source files)
src/01_download_data.py   downloads both datasets from Hugging Face and prepares the subsets
src/02_eda_reviews.py     exploratory analysis of dataset 1
src/03_eda_games.py       exploratory analysis of dataset 2
src/04_eda_joined.py      analysis of the joined datasets
src/common.py             shared helpers (paths, charts, statistical tests)
reports/stage1/           stage 1 report, figures (figures/) and numerical results (results/)
```

## How to reproduce the results

Requires Python 3.10+.

```
python -m pip install -r requirements.txt
python src/01_download_data.py      # optional – the prepared data is already in data/processed/
python src/02_eda_reviews.py
python src/03_eda_games.py
python src/04_eda_joined.py
```

`01_download_data.py` downloads about 200 MB into `data/raw/` (this directory is not committed) at the pinned commits listed above.
If downloading from Hugging Face does not work, download both files manually:

- https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset/resolve/8e9c6b569122fb20668f5da3b0f3970a853fc5d8/australian_user_reviews.json.gz
- https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/6a31fcb42082345acc817e176b28849b7c05f248/data/train-00000-of-00001.parquet

and run:

```
python src/01_download_data.py --reviews PATH/australian_user_reviews.json.gz --games PATH/train-00000-of-00001.parquet
```

Figures are saved to `reports/stage1/figures/` and numbers to `reports/stage1/results/*.json`.

## Key findings of stage 1

- 88.5% of reviews recommend the game, so the classes are strongly imbalanced. The share of "recommended" falls from 98.5% (2010) to 81.9% (2016).
- Negative reviews are longer (median 21 words vs 14) and more often give concrete reasons: updates, bugs, money and DLC.
- Among the 30,621 games with at least 50 ratings, the median share of positive ratings is 82%. Free games are rated lower than paid ones, and the genre matters (lowest: Free To Play and simulations).
- Ratings in both datasets agree (Spearman correlation 0.75 for 361 games). The largest differences are for games whose reception changed over time – e.g. No Man's Sky: 30% "recommended" in reviews from 2016, 83% positive ratings today.

Details and research questions: [reports/stage1/report_stage1.pdf](reports/stage1/report_stage1.pdf).
