"""
Collect human reviews on human-authored PRs for RQ2 trust analysis.

Samples merged/closed human PRs from top AI-active repos, fetches their
reviews via the GitHub API, and saves them in the same schema as the
existing human_reviews_{agent}.jsonl files.

Usage:
    python collect_human_pr_reviews.py --dry-run       # Preview sampling, no API calls
    python collect_human_pr_reviews.py                 # Run collection
    python collect_human_pr_reviews.py --verify-only   # Check per-agent repo coverage

Environment variables (from .env):
    GITHUB_API_TOKEN   - GitHub personal access token
    MIN_STARS          - Minimum repo stars (default 500)
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set
from urllib.parse import urlparse

from github import Github
from github.PaginatedList import PaginatedList
from github.PullRequestReview import PullRequestReview

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import (
    AI_PR_DIR,
    HUMAN_PR_DIR,
    HUMAN_REVIEW_DIR,
    RAW_DIR,
    AGENTS,
)
from scripts.utils.github_client import load_env, get_github_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Sampling
TOP_REPOS_COUNT = 100  # Number of top AI-active repos to sample from
TARGET_HUMAN_PRS = 7_500  # Target number of human PRs to sample
RANDOM_SEED = 42  # For reproducible sampling

# Output
OUTPUT_DIR = RAW_DIR / "human_reviews"
OUTPUT_FILE = OUTPUT_DIR / "human_reviews_on_human_prs.jsonl"
SAMPLED_PRS_FILE = (
    OUTPUT_DIR / "sampled_human_prs.jsonl"
)  # Track which PRs were sampled

# API settings
SECONDS_BETWEEN_REQUESTS = 0.15
CHECKPOINT_INTERVAL = 100  # Save progress log every N PRs

# Agents to skip in coverage verification (too few reviews)
SKIP_AGENTS = {"codegen"}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_existing_ids(path: Path) -> Set[str]:
    """Load IDs already collected for dedup."""
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def load_repos_json(path: Path) -> dict:
    """Load repos_with_500_stars.json."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run collection first")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ai_pr_repos_by_agent() -> Dict[str, Set[str]]:
    """Load which repos each AI agent is active in."""
    agent_repos: Dict[str, Set[str]] = {}
    for agent in AGENTS:
        if agent in SKIP_AGENTS:
            continue
        pr_file = AI_PR_DIR / f"ai_authored_{agent}.jsonl"
        if not pr_file.exists():
            continue
        repos = set()
        with open(pr_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    repo = entry.get("repo_full_name")
                    if repo:
                        repos.add(repo)
        agent_repos[agent] = repos
    return agent_repos


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def select_top_repos(top_n_global: int, top_n_per_agent: int = 30) -> list[str]:
    """
    Select repos using union of:
    1. Top N repos globally by AI PR count (for broad coverage)
    2. Top N repos per agent by that agent's PR count (for per-agent coverage)

    This ensures every agent - including low-volume ones like Jules and Devin -
    has meaningful repo overlap in the human PR sample.
    """
    repos = set()

    # Global top N from repos JSON
    repos_json_path = HUMAN_PR_DIR / "repos_with_500_stars.json"
    repos_json = load_repos_json(repos_json_path)
    global_sorted = sorted(
        repos_json.get("repos", []),
        key=lambda r: r.get("ai_pr_count", 0),
        reverse=True,
    )
    global_top = {r["name"] for r in global_sorted[:top_n_global]}
    repos.update(global_top)
    print(f"  Global top {top_n_global}: {len(global_top)} repos")

    # Per-agent top N
    for agent in AGENTS:
        if agent in SKIP_AGENTS:
            continue
        pr_file = AI_PR_DIR / f"ai_authored_{agent}.jsonl"
        if not pr_file.exists():
            continue

        # Count PRs per repo for this agent
        repo_counts: Dict[str, int] = {}
        with open(pr_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    repo = entry.get("repo_full_name")
                    if repo:
                        repo_counts[repo] = repo_counts.get(repo, 0) + 1

        agent_top = sorted(repo_counts, key=repo_counts.get, reverse=True)[
            :top_n_per_agent
        ]
        new_repos = set(agent_top) - repos
        repos.update(agent_top)
        print(f"  {agent:<15}: top {top_n_per_agent} repos, {len(new_repos)} new")

    print(f"  Total unique repos: {len(repos)}")
    return sorted(repos)


def sample_human_prs(
    top_repos: list[str],
    target_count: int,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Sample merged/closed human PRs from the top repos.

    Loads human_authored_prs.jsonl, filters to PRs in top_repos that
    are merged or closed (not open), then randomly samples up to
    target_count PRs.
    """
    human_pr_file = HUMAN_PR_DIR / "human_authored_prs.jsonl"
    all_prs = load_jsonl(human_pr_file)
    print(f"  Loaded {len(all_prs):,} total human PRs")

    top_repo_set = set(top_repos)

    # Filter: in top repos AND not open
    eligible = [
        pr
        for pr in all_prs
        if pr.get("repo_full_name") in top_repo_set and pr.get("state") != "open"
    ]
    print(f"  Eligible (in top repos, merged/closed): {len(eligible):,}")

    # Sample
    rng = random.Random(seed)
    if len(eligible) <= target_count:
        print(f"  Using all {len(eligible):,} eligible PRs (fewer than target {target_count:,})")
        sample = eligible
    else:
        sample = rng.sample(eligible, target_count)
        print(f"  Sampled {len(sample):,} PRs from {len(eligible):,} eligible")

    return sample


# ---------------------------------------------------------------------------
# Coverage Verification
# ---------------------------------------------------------------------------
def verify_agent_coverage(
    sampled_prs: list[dict],
    agent_repos: Dict[str, Set[str]],
) -> Dict[str, dict]:
    """Check how well the sample covers each agent's repos."""
    sampled_repos = set(pr.get("repo_full_name") for pr in sampled_prs)

    # Count human PRs per repo in sample
    repo_pr_counts = Counter(pr.get("repo_full_name") for pr in sampled_prs)

    results = {}
    print(f"\n{'Agent':<20} {'Agent Repos':>12} {'Overlap':>10} {'Coverage':>10} {'Human PRs in Overlap':>22}")
    print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*22}")

    for agent in sorted(agent_repos.keys()):
        if agent in SKIP_AGENTS:
            continue
        agent_repo_set = agent_repos[agent]
        overlap = agent_repo_set & sampled_repos
        overlap_pr_count = sum(repo_pr_counts.get(r, 0) for r in overlap)
        coverage_pct = 100 * len(overlap) / len(agent_repo_set) if agent_repo_set else 0

        results[agent] = {
            "agent_repos": len(agent_repo_set),
            "overlap": len(overlap),
            "coverage_pct": coverage_pct,
            "human_prs_in_overlap": overlap_pr_count,
        }

        flag = " [!]" if coverage_pct < 50 else ""
        print(
            f"  {agent:<18} {len(agent_repo_set):>10} {len(overlap):>10} "
            f"{coverage_pct:>9.1f}% {overlap_pr_count:>20}{flag}"
        )

    low_coverage = [a for a, r in results.items() if r["coverage_pct"] < 50]
    if low_coverage:
        print(f"\n  [!] Low coverage agents: {', '.join(low_coverage)}")
        print(f"      Consider supplementing with targeted samples from their repos.")

    return results


# ---------------------------------------------------------------------------
# Review Collection
# ---------------------------------------------------------------------------
def _get_author_association(review) -> str | None:
    """Extract author_association from a review, safely."""
    raw = getattr(review, "raw_data", None) or getattr(review, "_rawData", None)
    if isinstance(raw, dict):
        return raw.get("author_association")
    return None


def collect_reviews(
    client: Github,
    sampled_prs: list[dict],
    existing_review_ids: Set[str],
    existing_pr_ids_done: Set[str],
) -> dict:
    """
    Fetch reviews for sampled human PRs.

    Uses the same schema and filtering as collect_ai_human_prs_reviews.py:
    - Only human reviewers (user.type == "User")
    - Only reviews submitted before merge/close time
    - Same output fields

    Returns stats dict.
    """
    stats = {
        "prs_total": len(sampled_prs),
        "prs_already_done": 0,
        "prs_processed": 0,
        "prs_with_reviews": 0,
        "prs_failed": 0,
        "reviews_saved": 0,
        "reviews_skipped_bot": 0,
        "reviews_skipped_after_cutoff": 0,
        "reviews_skipped_no_user": 0,
        "reviews_duplicate": 0,
        "failed_repos": [],
    }

    start_time = time.time()

    # Filter out already-processed PRs
    remaining = [pr for pr in sampled_prs if str(pr["id"]) not in existing_pr_ids_done]
    stats["prs_already_done"] = len(sampled_prs) - len(remaining)

    if stats["prs_already_done"] > 0:
        print(f"  Skipping {stats['prs_already_done']:,} already-processed PRs")

    print(f"  Processing {len(remaining):,} PRs...")

    with (
        open(OUTPUT_FILE, "a", encoding="utf-8") as review_fh,
        open(SAMPLED_PRS_FILE, "a", encoding="utf-8") as pr_fh,
    ):
        for i, pr in enumerate(remaining, 1):
            try:
                repo_full_name = pr["repo_full_name"]
                pr_number = pr["number"]
                pr_id = pr["id"]

                # Determine review cutoff
                merged_at = pr.get("merged_at")
                closed_at = pr.get("closed_at")

                if merged_at:
                    review_cutoff = datetime.fromisoformat(merged_at)
                    if review_cutoff.tzinfo is None:
                        review_cutoff = review_cutoff.replace(tzinfo=timezone.utc)
                elif closed_at:
                    review_cutoff = datetime.fromisoformat(closed_at)
                    if review_cutoff.tzinfo is None:
                        review_cutoff = review_cutoff.replace(tzinfo=timezone.utc)
                else:
                    # No merge or close time - skip
                    stats["prs_processed"] += 1
                    continue

                # Fetch reviews via API
                repo = client.get_repo(repo_full_name)
                pull = repo.get_pull(pr_number)
                reviews = pull.get_reviews()

                pr_review_count = 0

                for review in reviews:
                    try:
                        if not review.user:
                            stats["reviews_skipped_no_user"] += 1
                            continue
                        if review.user.type != "User":
                            stats["reviews_skipped_bot"] += 1
                            continue
                        if review.submitted_at and review.submitted_at > review_cutoff:
                            stats["reviews_skipped_after_cutoff"] += 1
                            continue

                        review_id_str = str(review.id)
                        if review_id_str in existing_review_ids:
                            stats["reviews_duplicate"] += 1
                            continue

                        # Build review record - same schema as AI PR reviews
                        pr_api_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"

                        review_record = {
                            "id": review.id,
                            "url": f"{pr_api_url}/reviews/{review.id}",
                            "state": review.state,
                            "user_login": review.user.login,
                            "user_type": review.user.type,
                            "author_association": _get_author_association(review),
                            "target_commit": review.commit_id,
                            "submitted_at": (
                                review.submitted_at.isoformat()
                                if review.submitted_at
                                else None
                            ),
                            "pr_id": pr_id,
                            "pr_url": pr_api_url,
                            "html_url": review.html_url,
                            "body": review.body,
                        }

                        json.dump(review_record, review_fh)
                        review_fh.write("\n")
                        existing_review_ids.add(review_id_str)
                        stats["reviews_saved"] += 1
                        pr_review_count += 1

                    except Exception:
                        print(f"\n    Error on review for PR #{pr_number}: {traceback.format_exc()}")

                if pr_review_count > 0:
                    stats["prs_with_reviews"] += 1

                # Record that this PR was processed
                json.dump(
                    {
                        "id": pr_id,
                        "repo": repo_full_name,
                        "number": pr_number,
                        "reviews_collected": pr_review_count,
                    },
                    pr_fh,
                )
                pr_fh.write("\n")
                existing_pr_ids_done.add(str(pr_id))
                stats["prs_processed"] += 1

                # Progress logging
                if i % CHECKPOINT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_mins = (len(remaining) - i) / rate / 60 if rate > 0 else 0
                    print(
                        f"    [{i:,}/{len(remaining):,}] "
                        f"{stats['reviews_saved']:,} reviews saved, "
                        f"{rate:.1f} PRs/sec, "
                        f"ETA ~{eta_mins:.0f} min"
                    )
                    review_fh.flush()
                    pr_fh.flush()

            except Exception:
                stats["prs_failed"] += 1
                failed_repo = pr.get("repo_full_name", "unknown")
                if failed_repo not in stats["failed_repos"]:
                    stats["failed_repos"].append(failed_repo)
                if stats["prs_failed"] <= 10:
                    print(
                        f"\n    Error on PR {failed_repo}#{pr.get('number')}: {traceback.format_exc()}"
                    )
                elif stats["prs_failed"] == 11:
                    print("    (suppressing further error details)")

    elapsed = math.floor(time.time() - start_time)
    print(f"\n  Collection complete in {elapsed}s")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dry_run = "--dry-run" in sys.argv
    verify_only = "--verify-only" in sys.argv

    print("=" * 70)
    print("Human PR Review Collection for RQ2")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Select top repos
    # ------------------------------------------------------------------
    top_repos = select_top_repos(
        top_n_global=TOP_REPOS_COUNT,
        top_n_per_agent=30,
    )

    print(f"\nStep 1: Selected {len(top_repos)} repos (global top {TOP_REPOS_COUNT} + top 30 per agent)")
    print(f"  First 5: {', '.join(top_repos[:5])}")

    # ------------------------------------------------------------------
    # Step 2: Sample human PRs
    # ------------------------------------------------------------------
    print(f"\nStep 2: Sampling human PRs")
    sampled_prs = sample_human_prs(top_repos, TARGET_HUMAN_PRS)

    # Distribution across repos
    repo_dist = Counter(pr["repo_full_name"] for pr in sampled_prs)
    print(f"  Spanning {len(repo_dist)} repos")
    print(f"  Top 5 repos by sample count:")
    for repo, count in repo_dist.most_common(5):
        print(f"    {repo}: {count}")

    # Merged vs. closed breakdown
    merged = sum(1 for pr in sampled_prs if pr.get("merged_at"))
    closed_only = len(sampled_prs) - merged
    print(f"  Merged: {merged:,}, Closed (not merged): {closed_only:,}")

    # ------------------------------------------------------------------
    # Step 3: Verify per-agent coverage
    # ------------------------------------------------------------------
    print(f"\nStep 3: Per-agent repo coverage")
    agent_repos = load_ai_pr_repos_by_agent()
    coverage = verify_agent_coverage(sampled_prs, agent_repos)

    if verify_only:
        print("\n[VERIFY ONLY] Done.")
        return

    if dry_run:
        # Estimate API calls
        est_calls = len(sampled_prs) * 3  # get_repo + get_pull + get_reviews per PR
        est_hours = est_calls / 5000  # GitHub rate limit: 5,000 requests/hour
        print(f"\n[DRY RUN] Would collect reviews for {len(sampled_prs):,} PRs")
        print(f"  Estimated API calls: ~{est_calls:,}")
        print(f"  Estimated time: ~{est_hours:.1f} hours (GitHub rate limit: 5k/hr)")
        print(f"  Output: {OUTPUT_FILE}")
        print("\n  Run without --dry-run to collect.")
        return

    # ------------------------------------------------------------------
    # Step 4: Collect reviews
    # ------------------------------------------------------------------
    print(f"\nStep 4: Collecting reviews")

    api_token = load_env()
    client = get_github_client(
        api_token, seconds_between_requests=SECONDS_BETWEEN_REQUESTS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_review_ids = load_existing_ids(OUTPUT_FILE)
    existing_pr_ids_done = load_existing_ids(SAMPLED_PRS_FILE)
    print(f"  Existing reviews on disk: {len(existing_review_ids):,}")
    print(f"  Already-processed PRs: {len(existing_pr_ids_done):,}")

    stats = collect_reviews(
        client=client,
        sampled_prs=sampled_prs,
        existing_review_ids=existing_review_ids,
        existing_pr_ids_done=existing_pr_ids_done,
    )

    client.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    for key, val in stats.items():
        if key == "failed_repos":
            continue
        print(f"  {key:<35} {val:>10,}")

    if stats["failed_repos"]:
        print(f"\n  Failed repos ({len(stats['failed_repos'])}):")
        for repo in stats["failed_repos"]:
            print(f"    - {repo}")

    print(f"\n  Output files:")
    print(f"    Reviews: {OUTPUT_FILE}")
    print(f"    Processed PRs: {SAMPLED_PRS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
