"""Analysis helper for local PR classification data."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats
from cliffs_delta import cliffs_delta

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.utils.config import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    AGENTS,
    AI_PR_DIR,
    HUMAN_PR_DIR,
)

# Directories
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CLASSIFIED_DIR = PROCESSED_DIR / "classified_prs"

COLOR_MAP = {
    "Human": "#56B4E9",
    "codegen": "#E69F00",
    "Codegen": "#E69F00",
    "devin": "#009E73",
    "Devin": "#009E73",
    "copilot": "#0072B2",
    "GitHub Copilot": "#0072B2",
    "cursor": "#785EF0",
    "Cursor": "#785EF0",
    "jules": "#F0E442",
    "Google Labs Jules": "#F0E442",
    "claude-code": "#DC267F",
    "Claude Code": "#DC267F",
    "codex": "#D55E00",
    "OpenAI Codex": "#D55E00",
}

NAME_MAPPING = {
    "codegen": "Codegen",
    "devin": "Devin",
    "copilot": "GitHub Copilot",
    "cursor": "Cursor",
    "jules": "Google Labs Jules",
    "claude-code": "Claude Code",
    "codex": "OpenAI Codex",
    "Human": "Human",
}

for key, value in NAME_MAPPING.items():
    if key not in COLOR_MAP:
        COLOR_MAP[key] = COLOR_MAP.get(value, "#444444")

FLOW_ORDER = [
    "feat",
    "fix",
    "perf",
    "refactor",
    "style",
    "docs",
    "test",
    "chore",
    "build",
    "ci",
    "other",
]


# -----------------------------------------------------------------------------
# Data Container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalData:
    """Container for local PR and classification data."""

    pr_df: pd.DataFrame
    lbl_df: pd.DataFrame


# -----------------------------------------------------------------------------
# JSONL Loading Functions
# -----------------------------------------------------------------------------
def load_jsonl_to_df(file_path: Path) -> pd.DataFrame:
    """Load JSONL file into pandas DataFrame."""
    if not file_path.exists():
        return pd.DataFrame()

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records) if records else pd.DataFrame()

    # Standardize ID column to string if it exists
    if not df.empty and "id" in df.columns:
        df["id"] = df["id"].astype(str)

    return df


def load_all_prs() -> pd.DataFrame:
    """Load all PRs from JSONL files."""
    all_prs = []

    # Load AI PRs
    for agent in AGENTS:
        pr_file = AI_PR_DIR / f"ai_authored_{agent}.jsonl"
        if pr_file.exists():
            df = load_jsonl_to_df(pr_file)
            if not df.empty:
                df["agent"] = agent
                all_prs.append(df)

    # Load Human PRs
    human_pr_file = HUMAN_PR_DIR / "human_authored_prs.jsonl"
    if human_pr_file.exists():
        df = load_jsonl_to_df(human_pr_file)
        if not df.empty:
            df["agent"] = "Human"
            all_prs.append(df)

    if not all_prs:
        return pd.DataFrame()

    # Combine all
    pr_df = pd.concat(all_prs, ignore_index=True)

    # Convert datetime columns
    for col in ["created_at", "closed_at", "merged_at"]:
        if col in pr_df.columns:
            pr_df[col] = pd.to_datetime(pr_df[col], utc=True, errors="coerce")

    return pr_df


def load_all_classifications() -> pd.DataFrame:
    """Load all classifications from JSONL files."""
    all_classifications = []

    # Load all agent classifications
    for agent in AGENTS + ["Human"]:
        class_file = CLASSIFIED_DIR / f"{agent}_pr_task_type.jsonl"
        if class_file.exists():
            df = load_jsonl_to_df(class_file)
            if not df.empty:
                all_classifications.append(df)

    if not all_classifications:
        return pd.DataFrame()

    # Combine all
    class_df = pd.concat(all_classifications, ignore_index=True)

    # Clean type column
    if "type" in class_df.columns:
        class_df["type"] = class_df["type"].astype(str).str.strip()

    return class_df


# -----------------------------------------------------------------------------
# Data Loading (Main Entry Point)
# -----------------------------------------------------------------------------
def load_local_data(
    stars_range: tuple[float | None, float | None] | None = None,
) -> LocalData:
    """
    Load local PR and classification data from JSONL/CSV files.

    Args:
        stars_range: Optional (min_stars, max_stars) filter.

    Returns:
        LocalData containing pr_df and lbl_df
    """
    pr_df = load_all_prs()
    lbl_df = load_all_classifications()

    if pr_df.empty:
        raise FileNotFoundError(
            "No PR data found. Check that files exist in:\n"
            f"  - {AI_PR_DIR}/ai_authored_*.jsonl\n"
            f"  - {HUMAN_PR_DIR}/human_authored_prs.jsonl"
        )

    if lbl_df.empty:
        raise FileNotFoundError(
            f"No classification data found in {CLASSIFIED_DIR}/\n"
            "Run classify_prs.py first to generate *_pr_task_type.jsonl files."
        )

    # Apply star filtering if requested
    if stars_range is not None:
        min_stars, max_stars = stars_range

        if "repo_stars" in pr_df.columns:
            mask = pr_df["repo_stars"].notna()

            if min_stars is not None:
                mask &= pr_df["repo_stars"] >= min_stars
            if max_stars is not None:
                mask &= pr_df["repo_stars"] <= max_stars

            pr_df = pr_df[mask].copy()

            # Filter labels to match
            if "id" in lbl_df.columns and "id" in pr_df.columns:
                keep_ids = set(pr_df["id"].unique())
                lbl_df = lbl_df[lbl_df["id"].isin(keep_ids)].copy()

    return LocalData(pr_df=pr_df, lbl_df=lbl_df)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_prs(agent: str, data: LocalData) -> pd.DataFrame:
    """Get PRs for a specific agent."""
    return data.pr_df[data.pr_df["agent"] == agent].copy()


def load_labels(agent: str, data: LocalData) -> pd.DataFrame:
    """Get classifications for a specific agent."""
    return data.lbl_df[data.lbl_df["agent"] == agent].copy()


def get_agents(data: LocalData) -> list[str]:
    """Get list of all agents with data."""
    return sorted(data.pr_df["agent"].dropna().unique().tolist())


# -----------------------------------------------------------------------------
# Statistical Tests
# -----------------------------------------------------------------------------
def mannUandCliffdelta(dist1, dist2):
    """Perform Mann-Whitney U test and Cliff's delta effect size."""
    d, size = cliffs_delta(dist1, dist2)
    print(f"Cliff's delta: {size}, d={d}")

    u, p = stats.mannwhitneyu(dist1, dist2, alternative="two-sided")
    print(f"Mann-Whitney-U-test: u={u} p={p}")

    return u, p, d, size


# -----------------------------------------------------------------------------
# Agent Name Normalization
# -----------------------------------------------------------------------------
def normalize_agent_name(agent: str) -> str:
    """Convert internal agent name to display name."""
    return NAME_MAPPING.get(agent, agent)


def get_agent_color(agent: str) -> str:
    """Get color for agent."""
    return COLOR_MAP.get(agent, COLOR_MAP.get(normalize_agent_name(agent), "#444444"))


# -----------------------------------------------------------------------------
# Data Validation
# -----------------------------------------------------------------------------
def validate_data(data: LocalData) -> None:
    """Validate that loaded data has expected structure."""
    required_pr_cols = ["id", "agent", "title", "state", "created_at"]
    required_lbl_cols = ["id", "agent", "type"]

    missing_pr = [col for col in required_pr_cols if col not in data.pr_df.columns]
    missing_lbl = [col for col in required_lbl_cols if col not in data.lbl_df.columns]

    if missing_pr:
        raise ValueError(f"PR data missing columns: {missing_pr}")
    if missing_lbl:
        raise ValueError(f"Classification data missing columns: {missing_lbl}")

    print(f"✓ Data validated")
    print(f"  PRs: {len(data.pr_df)} across {data.pr_df['agent'].nunique()} agents")
    print(f"  Classifications: {len(data.lbl_df)}")
    print(f"  Agents: {', '.join(get_agents(data))}")


# -----------------------------------------------------------------------------
# Quick Stats
# -----------------------------------------------------------------------------
def print_summary_stats(data: LocalData) -> None:
    """Print summary statistics about the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    agents = get_agents(data)

    for agent in agents:
        prs = load_prs(agent, data)
        labels = load_labels(agent, data)

        total = len(prs)
        classified = len(labels)
        merged = prs["merged_at"].notna().sum() if "merged_at" in prs.columns else 0
        closed = (
            (prs["state"] == "closed")
            & (prs["merged_at"].isna() if "merged_at" in prs.columns else True)
        ).sum()
        open_count = (prs["state"] == "open").sum() if "state" in prs.columns else 0

        print(f"\n{normalize_agent_name(agent)}:")
        print(f"  Total PRs: {total}")
        if total > 0:
            print(f"  Classified: {classified} ({100*classified/total:.1f}%)")
            print(f"  Merged: {merged} ({100*merged/total:.1f}%)")
        print(f"  Closed (not merged): {closed}")
        print(f"  Open: {open_count}")

        if not labels.empty:
            type_dist = labels["type"].value_counts().head(5)
            print(f"  Top types: {', '.join([f'{t}({c})' for t, c in type_dist.items()])}")

    print("\n" + "=" * 60)
