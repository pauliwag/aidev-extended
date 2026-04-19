# aidev-extended

Replication package for a longitudinal study of AI coding agent pull requests on GitHub (Nov 2025 – Feb 2026), extending the AIDev dataset.

## Data

Raw and processed data ship as zipped archives under `data/archives/` (tracked with Git LFS). Unzip before running anything:

```bash
cd data
unzip archives/raw.zip
unzip archives/processed.zip
```

Expected layout after unzip:

```
data/raw/ai_authored_prs/ai_authored_{agent}.jsonl
data/raw/ai_authored_prs/repos_with_500_stars.json
data/raw/human_authored_prs/human_authored_prs.jsonl
data/raw/human_reviews/human_reviews_{agent}.jsonl
data/raw/human_reviews/human_reviews_on_human_prs.jsonl
data/raw/human_reviews/sampled_human_prs.jsonl
data/processed/classified_prs/{agent}_pr_task_type.jsonl
data/processed/excluded_human_prs/excluded_human_prs.jsonl
```

## Setup

```bash
make env
conda activate agentic-prs
cp .env.example .env   # only needed if re-running collection
```

Classification also requires [Ollama](https://ollama.com) with `qwen3:30b-instruct` pulled.

## Reproduce the analysis

With the data unzipped, run the notebooks in order:

1. `notebooks/00_pr_analysis.ipynb` – RQ1 (contribution profiles, acceptance rates, turnaround)
2. `notebooks/02_review_trust_analysis.ipynb` – RQ2 (review depth, trust patterns)

Figures and LaTeX tables are written to `figures/`.

## Re-run collection from scratch

Requires a GitHub personal access token in `.env` and takes ~1.5 weeks of wall time.

```bash
# Set COLLECT_FROM_DATE and COLLECT_TO_DATE in .env, then:
python scripts/collection/collect_ai_human_prs_reviews.py
python scripts/collection/collect_human_pr_reviews.py

# After combining language splits:
jupyter nbconvert --execute notebooks/01_combine_agent_splits.ipynb

# Clean human PRs of bot/leaked-AI entries:
python scripts/processing/clean_human_prs.py --discover   # inspect
python scripts/processing/clean_human_prs.py              # apply

# Classify:
python scripts/classification/classify_prs.py
```

## Layout

```
data/              Zipped archives (LFS) + pr_arena tracking data
notebooks/         Analysis notebooks and helpers
scripts/
  collection/      GitHub API collection
  processing/      Human PR cleaning
  classification/  Conventional Commits classification (regex + LLM)
  utils/           Shared config and GitHub client
figures/           Generated figures and tables
```