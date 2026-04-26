# aidev-extended

Replication package for a longitudinal study of AI coding agent pull requests on GitHub (Nov 2025 – Feb 2026), extending the [AIDev dataset](https://huggingface.co/datasets/hao-li/AIDev).

The study covers **209,934 PRs** and **65,924 code reviews** across **5,362 repositories** (≥500 GitHub stars), comparing six AI coding agents—Claude Code, OpenAI Codex, GitHub Copilot, Cursor, Devin, and Google Labs Jules—against a human-authored baseline.

## Layout

```
data/
  archives/             Zipped raw + processed data (Git LFS)
  pr_arena/             PR Arena tracking data
notebooks/
  01_combine_agent_splits.ipynb   Merge per-language Phase 1 partitions
  02_pr_analysis.ipynb            RQ1: acceptance, turnaround, task types
  03_review_trust_analysis.ipynb  RQ2: review depth and trust patterns
  analysis_helper.py              Shared loading / plotting / stats
scripts/
  collection/           GitHub API collection (Phases 1–3)
  processing/           Human PR cleaning (bot accounts, AI leaks)
  classification/       Conventional Commits classification (regex + LLM)
  utils/                Shared config and GitHub client
figures/                Generated figures, tables, and CSVs
```

## Data

Raw and processed data ship as Git LFS archives (~114 MB total):

- `data/archives/raw.zip` – AI-authored PRs, human-authored PRs, human reviews, sampled PRs
- `data/archives/processed.zip` – Conventional Commits classifications, quarantined PRs

Unzipping recreates `data/raw/` and `data/processed/` in place. Expected layout after unzip:

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
conda activate aidev-extended
cp .env.example .env   # only needed if re-running collection
```

Classification also requires [Ollama](https://ollama.com) with `qwen3:30b-instruct` pulled.

## Reproduce the Analysis

With the data unzipped, run the notebooks in order:

1. `notebooks/02_pr_analysis.ipynb` – RQ1 (contribution profiles, acceptance rates, turnaround)
2. `notebooks/03_review_trust_analysis.ipynb` – RQ2 (review depth, trust patterns)

Generated figures and LaTeX tables land in `figures/`.

## Re-run Collection from Scratch

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

# Classify with regex + LLM:
python scripts/classification/classify_prs.py
```

## Citation

See the accompanying paper for methodology and full results. AIDev, which this work extends:

```bibtex
@misc{li2025aiteammates,
      title={The Rise of AI Teammates in Software Engineering (SE) 3.0: How Autonomous Coding Agents Are Reshaping Software Engineering}, 
      author={Hao Li and Haoxiang Zhang and Ahmed E. Hassan},
      year={2025},
      eprint={2507.15003},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2507.15003}, 
}
```