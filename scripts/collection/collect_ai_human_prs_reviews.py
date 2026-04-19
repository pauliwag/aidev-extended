import io
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse

from github import Github
from github.PaginatedList import PaginatedList
from github.PullRequestReview import PullRequestReview

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import AI_PR_DIR, HUMAN_REVIEW_DIR, HUMAN_PR_DIR, MIN_STARS
from scripts.utils.github_client import load_env, get_github_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BODY_MAX_CHARS = 4_000
HUMAN_PR_WEEKLY_CAP = 10_000  # Max human PRs collected per calendar week

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(d: date) -> datetime:
    """Promote a date to a timezone-aware datetime at midnight UTC."""
    if isinstance(d, datetime):
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def iter_date_chunks(
    start: datetime, end: datetime, chunk_days: float
):
    """Yield ``(chunk_start, chunk_end)`` pairs walking **backwards** from *end*."""
    delta = timedelta(days=chunk_days)
    cursor = end
    while cursor > start:
        chunk_end = cursor
        chunk_start = max(cursor - delta, start)
        yield chunk_start, chunk_end
        cursor = chunk_start


def iter_weekly_windows(start: date, end: date):
    """Yield ``(week_start, week_end)`` date pairs covering the full range."""
    cursor = start
    while cursor < end:
        week_end = min(cursor + timedelta(days=7), end)
        yield cursor, week_end
        cursor = week_end


def fmt_dt(dt: datetime) -> str:
    """Format a datetime for the GitHub search API (ISO-8601 / UTC)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def truncate_body(body: str | None) -> str | None:
    if body is None:
        return None
    if len(body) <= BODY_MAX_CHARS:
        return body
    return body[:BODY_MAX_CHARS] + f"\n[truncated at {BODY_MAX_CHARS} chars]"


def load_existing_ids(path: Path) -> Set[str]:
    """Read a JSONL file and return the set of ``id`` values already present."""
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def load_repos_json(path: Path) -> dict:
    """Load the existing repos-with-stars JSON, or return an empty skeleton."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"count": 0, "runs": [], "repos": []}


def merge_repos_json(
    existing: dict,
    repos_with_stars: Set[str],
    repo_star_cache: Dict[str, int],
    repo_ai_pr_count: Dict[str, int],
    date_range: Tuple[date, date],
) -> dict:
    """Merge newly discovered repos into the existing JSON structure."""
    # Index existing repos by name for fast lookup
    by_name: Dict[str, dict] = {}
    for entry in existing.get("repos", []):
        by_name[entry["name"]] = entry

    # Upsert new repos
    for repo in repos_with_stars:
        stars = repo_star_cache.get(repo, 0)
        ai_count = repo_ai_pr_count.get(repo, 0)
        if repo in by_name:
            by_name[repo]["stars"] = max(by_name[repo].get("stars", 0), stars)
            by_name[repo]["ai_pr_count"] = by_name[repo].get("ai_pr_count", 0) + ai_count
        else:
            by_name[repo] = {"name": repo, "stars": stars, "ai_pr_count": ai_count}

    repos_sorted = sorted(by_name.values(), key=lambda r: r["stars"], reverse=True)

    runs = existing.get("runs", [])
    runs.append({
        "from": date_range[0].isoformat(),
        "to": date_range[1].isoformat(),
        "new_repos": len(repos_with_stars - set(e["name"] for e in existing.get("repos", []))),
        "total_ai_prs_this_run": sum(repo_ai_pr_count.values()),
    })

    return {
        "count": len(repos_sorted),
        "runs": runs,
        "star_range": {
            "min": min((r["stars"] for r in repos_sorted), default=0),
            "max": max((r["stars"] for r in repos_sorted), default=0),
        },
        "repos": repos_sorted,
    }


def _get_author_association(review) -> str | None:
    """Extract author_association from a PullRequestReview, safely."""
    # PyGithub >=2.x exposes raw_data; fall back to _rawData for older versions
    raw = getattr(review, "raw_data", None) or getattr(review, "_rawData", None)
    if isinstance(raw, dict):
        return raw.get("author_association")
    return None

# ---------------------------------------------------------------------------
# Phase 1 - AI-authored PRs + human reviews
# ---------------------------------------------------------------------------

def collect_ai_authored_prs(
    client: Github,
    pr_file: io.FileIO,
    review_file: io.FileIO,
    agent_name: str,
    search_query: str,
    chunk_days: float,
    date_range: Tuple[date, date],
    repos_with_stars: Set[str],
    repo_star_cache: Dict[str, int],
    repo_ai_pr_count: Dict[str, int],
    existing_pr_ids: Set[str],
    existing_review_ids: Set[str],
):
    """
    Collect AI-authored PRs and their human reviews.

    Now supports sub-day chunking (``chunk_days`` may be < 1), collects PR
    body (truncated), gathers reviews on both merged **and** closed PRs,
    and records ``author_association`` on every review.
    """
    stats = {
        "pr_total_seen": 0,
        "pr_duplicate_skipped": 0,
        "pr_below_star_threshold": 0,
        "pr_copilot_wrong_user": 0,
        "pr_with_reviews_collected": 0,
        "pr_review_missing_info": 0,
        "pr_review_after_cutoff": 0,
        "pr_saved": 0,
        "human_review_total": 0,
        "repos_tracked": 0,
    }

    start_time = time.time()
    print(f"Searching for {agent_name} (chunk_days={chunk_days})")

    global_from = _dt(date_range[0])
    global_to = _dt(date_range[1])

    for chunk_start, chunk_end in iter_date_chunks(global_from, global_to, chunk_days):
        from_str = fmt_dt(chunk_start)
        to_str = fmt_dt(chunk_end)

        query = f"{search_query} created:{from_str}..{to_str}"
        pr_iterator = client.search_issues(query, sort="created", order="desc")

        label = f"{chunk_start.strftime('%Y-%m-%d %H:%M')}-{chunk_end.strftime('%m-%d %H:%M')}"
        print(f"\t{label}", end=" ")
        page_count = 0

        for pr in pr_iterator:
            try:
                page_count += 1
                stats["pr_total_seen"] += 1

                pr_id_str = str(pr.id)
                if pr_id_str in existing_pr_ids:
                    stats["pr_duplicate_skipped"] += 1
                    continue

                # Repo star filter 
                pr_url = pr.pull_request.url
                _, repo_owner, repo_name, _, _ = (
                    urlparse(pr_url).path.strip("/").split("/")
                )
                repo_full_name = f"{repo_owner}/{repo_name}"

                if repo_full_name not in repo_star_cache:
                    repo = client.get_repo(repo_full_name)
                    repo_star_cache[repo_full_name] = repo.stargazers_count

                stars = repo_star_cache[repo_full_name]
                if stars < MIN_STARS:
                    stats["pr_below_star_threshold"] += 1
                    continue

                # Track repo
                if repo_full_name not in repos_with_stars:
                    repos_with_stars.add(repo_full_name)
                    stats["repos_tracked"] += 1

                # Copilot author verification
                if agent_name.startswith("copilot-") and pr.user.login != "Copilot":
                    stats["pr_copilot_wrong_user"] += 1
                    continue

                # Track AI PR count for this repo
                repo_ai_pr_count[repo_full_name] = (
                    repo_ai_pr_count.get(repo_full_name, 0) + 1
                )

                # Collect reviews (merged OR closed)
                pr_reviews: list[dict] = []
                review_cutoff = None
                if pr.pull_request.merged_at:
                    review_cutoff = pr.pull_request.merged_at
                elif pr.closed_at:
                    review_cutoff = pr.closed_at

                if review_cutoff is not None:
                    reviews = PaginatedList(
                        PullRequestReview,
                        pr.requester,
                        f"{pr_url}/reviews",
                        None,
                    )
                    for review in reviews:
                        try:
                            if not review.user:
                                stats["pr_review_missing_info"] += 1
                                continue
                            if review.user.type != "User":
                                continue
                            if review.submitted_at > review_cutoff:
                                stats["pr_review_after_cutoff"] += 1
                                continue

                            review_id_str = str(review.id)
                            if review_id_str in existing_review_ids:
                                continue

                            pr_reviews.append({
                                "id": review.id,
                                "url": f"{pr_url}/reviews/{review.id}",
                                "state": review.state,
                                "user_login": review.user.login,
                                "user_type": review.user.type,
                                "author_association": _get_author_association(review),
                                "target_commit": review.commit_id,
                                "submitted_at": review.submitted_at.isoformat(),
                                "pr_id": pr.id,
                                "pr_url": review.pull_request_url,
                                "html_url": review.html_url,
                                "body": review.body,
                            })
                            existing_review_ids.add(review_id_str)
                        except Exception:
                            print(f"\n\t\tError on review: {pr.url}")
                            print(traceback.format_exc())

                if pr_reviews:
                    stats["pr_with_reviews_collected"] += 1
                    stats["human_review_total"] += len(pr_reviews)

                # Write PR
                pr_output = {
                    "id": pr.id,
                    "url": pr.url,
                    "repo_owner": repo_owner,
                    "repo_name": repo_name,
                    "repo_full_name": repo_full_name,
                    "repo_stars": stars,
                    "number": pr.number,
                    "title": pr.title,
                    "body": truncate_body(pr.body),
                    "author": pr.user.login,
                    "author_type": pr.user.type,
                    "state": pr.state,
                    "comments": pr.comments,
                    "human_reviews": len(pr_reviews),
                    "html_url": pr.pull_request.html_url,
                    "repo_url": pr.repository_url,
                    "created_at": pr.created_at.isoformat(),
                    "merged_at": (
                        pr.pull_request.merged_at.isoformat()
                        if pr.pull_request.merged_at
                        else None
                    ),
                    "closed_at": (
                        pr.closed_at.isoformat() if pr.closed_at else None
                    ),
                }

                json.dump(pr_output, pr_file)
                pr_file.write("\n")
                pr_file.flush()
                existing_pr_ids.add(pr_id_str)
                stats["pr_saved"] += 1

                for rev in pr_reviews:
                    json.dump(rev, review_file)
                    review_file.write("\n")
                review_file.flush()

            except Exception:
                print(f"\n\t\tError on PR: {pr.url}")
                print(traceback.format_exc())

        print(f"({page_count} results)")
        if page_count >= 1000:
            print(f"    WARNING: Hit 1000-result cap - tighten partitioning for {agent_name}")

    elapsed = math.floor(time.time() - start_time)
    print(f"Processed {agent_name} in {elapsed}s  |  saved {stats['pr_saved']} PRs, {stats['human_review_total']} reviews")
    print(f"  {stats}\n")
    return stats


# ---------------------------------------------------------------------------
# Phase 2 - Human-authored PRs (weekly-capped, append-safe)
# ---------------------------------------------------------------------------

def collect_human_authored_prs(
    client: Github,
    pr_file: io.FileIO,
    repos_with_stars: Set[str],
    date_range: Tuple[date, date],
    existing_pr_ids: Set[str],
    weekly_cap: int = HUMAN_PR_WEEKLY_CAP,
):
    """
    Collect human-authored PRs from repos discovered in Phase 1.

    The full date range is split into **weekly windows**.  Within each
    window the repo list is shuffled and collection stops once
    ``weekly_cap`` PRs have been saved, distributing coverage evenly
    across the month and reducing recency bias.
    """
    stats = {
        "pr_total_seen": 0,
        "pr_duplicate_skipped": 0,
        "pr_not_user_author": 0,
        "pr_saved": 0,
        "repos_processed": 0,
        "repos_failed": 0,
        "weeks_processed": 0,
        "weeks_capped": 0,
    }

    start_time = time.time()
    global_from, global_to = date_range

    weeks = list(iter_weekly_windows(global_from, global_to))
    print(
        f"Phase 2: Collecting human PRs from {len(repos_with_stars)} repos "
        f"across {len(weeks)} weekly windows  (cap={weekly_cap:,}/week)"
    )

    repo_list = sorted(repos_with_stars)

    for week_start, week_end in weeks:
        stats["weeks_processed"] += 1
        week_count = 0
        shuffled = list(repo_list)
        random.shuffle(shuffled)

        ws = week_start.isoformat()
        we = week_end.isoformat()
        print(f"\n  Week {ws} -> {we}")

        for repo_full_name in shuffled:
            if week_count >= weekly_cap:
                break

            try:
                stats["repos_processed"] += 1

                query = (
                    f"is:pr repo:{repo_full_name} "
                    f"created:{ws}..{we}"
                )
                pr_iterator = client.search_issues(
                    query, sort="created", order="desc"
                )

                for pr in pr_iterator:
                    if week_count >= weekly_cap:
                        break
                    try:
                        stats["pr_total_seen"] += 1

                        pr_id_str = str(pr.id)
                        if pr_id_str in existing_pr_ids:
                            stats["pr_duplicate_skipped"] += 1
                            continue

                        if pr.user.type != "User":
                            stats["pr_not_user_author"] += 1
                            continue
                        if pr.performed_via_github_app is not None:
                            stats["pr_not_user_author"] += 1
                            continue

                        pr_url = pr.pull_request.url
                        _, repo_owner, repo_name, _, _ = (
                            urlparse(pr_url).path.strip("/").split("/")
                        )

                        pr_output = {
                            "id": pr.id,
                            "url": pr.url,
                            "repo_owner": repo_owner,
                            "repo_name": repo_name,
                            "repo_full_name": repo_full_name,
                            "number": pr.number,
                            "title": pr.title,
                            "body": truncate_body(pr.body),
                            "author": pr.user.login,
                            "author_type": pr.user.type,
                            "state": pr.state,
                            "comments": pr.comments,
                            "html_url": pr.pull_request.html_url,
                            "repo_url": pr.repository_url,
                            "created_at": pr.created_at.isoformat(),
                            "merged_at": (
                                pr.pull_request.merged_at.isoformat()
                                if pr.pull_request.merged_at
                                else None
                            ),
                            "closed_at": (
                                pr.closed_at.isoformat()
                                if pr.closed_at
                                else None
                            ),
                        }

                        json.dump(pr_output, pr_file)
                        pr_file.write("\n")
                        pr_file.flush()
                        existing_pr_ids.add(pr_id_str)
                        stats["pr_saved"] += 1
                        week_count += 1

                    except Exception:
                        print(f"\n\t\tError on PR: {pr.url}")
                        print(traceback.format_exc())

            except Exception:
                stats["repos_failed"] += 1
                print(f"\n\t\tError on repo: {repo_full_name}")
                print(traceback.format_exc())

        if week_count >= weekly_cap:
            stats["weeks_capped"] += 1
        print(f"    -> {week_count:,} PRs collected (cap {'hit' if week_count >= weekly_cap else 'not hit'})")

    elapsed = math.floor(time.time() - start_time)
    print(f"\nPhase 2 done in {elapsed}s")
    print(f"  {stats}\n")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_token = load_env()

    from_date = datetime.strptime(
        os.getenv("COLLECT_FROM_DATE", "2025-11-01"), "%Y-%m-%d"
    ).date()
    to_date = datetime.strptime(
        os.getenv("COLLECT_TO_DATE", "2025-12-01"), "%Y-%m-%d"
    ).date()
    print(f"Running from {from_date} to {to_date}\n")

    client = get_github_client(api_token, seconds_between_requests=0.15)

    # ------------------------------------------------------------------
    # Agent configurations: (name, search_query, chunk_days)
    # ------------------------------------------------------------------
    agents: List[Tuple[str, str, float]] = [
        ("copilot-typescript", "is:pr head:copilot/ language:typescript", 0.3),  
        ("copilot-csharp", "is:pr head:copilot/ language:c#", 1),                
        ("copilot-python", "is:pr head:copilot/ language:python", 0.3),          
        ("copilot-go", "is:pr head:copilot/ language:go", 1),                     
        ("copilot-javascript", "is:pr head:copilot/ language:javascript", 0.5),   
        ("copilot-rust", "is:pr head:copilot/ language:rust", 1),                 
        ("copilot-cpp", "is:pr head:copilot/ language:c++", 1),                  
        ("copilot-java", "is:pr head:copilot/ language:java", 2),                
        ("copilot-php", "is:pr head:copilot/ language:php", 3),                   
        ("copilot-c", "is:pr head:copilot/ language:c", 3),                       
        ("copilot-ruby", "is:pr head:copilot/ language:ruby", 5),                
        ("copilot-html", "is:pr head:copilot/ language:html", 1),              
        ("copilot-powershell", "is:pr head:copilot/ language:powershell", 5),      
        ("copilot-kotlin", "is:pr head:copilot/ language:kotlin", 5),            
        ("copilot-other", "is:pr head:copilot/ -language:typescript -language:c# -language:python -language:go -language:javascript -language:rust -language:c++ -language:java -language:php -language:c -language:ruby -language:html -language:powershell -language:kotlin", 0.5),  # 645/day
        
        ("codex-python", "is:pr head:codex/ language:python", 0.2),     
        ("codex-typescript", "is:pr head:codex/ language:typescript", 0.2),  
        ("codex-go", "is:pr head:codex/ language:go", 0.5),                  
        ("codex-javascript", "is:pr head:codex/ language:javascript", 0.2), 
        ("codex-rust", "is:pr head:codex/ language:rust", 1),
        ("codex-cpp", "is:pr head:codex/ language:c++", 1),
        ("codex-csharp", "is:pr head:codex/ language:c#", 1),
        ("codex-java", "is:pr head:codex/ language:java", 2),
        ("codex-php", "is:pr head:codex/ language:php", 1),
        ("codex-c", "is:pr head:codex/ language:c", 3),
        ("codex-kotlin", "is:pr head:codex/ language:kotlin", 3),
        ("codex-ruby", "is:pr head:codex/ language:ruby", 5),
        ("codex-swift", "is:pr head:codex/ language:swift", 5),
        ("codex-other", "is:pr head:codex/ -language:python -language:typescript -language:go -language:javascript -language:rust -language:c++ -language:c# -language:java -language:php -language:c -language:kotlin -language:ruby -language:swift", 0.2),

        ("cursor-typescript", "is:pr head:cursor/ language:typescript", 1),      
        ("cursor-other", "is:pr head:cursor/ -language:typescript", 1),         
        
        ("jules-typescript", "is:pr author:google-labs-jules[bot] language:typescript", 0.3), 
        ("jules-python", "is:pr author:google-labs-jules[bot] language:python", 0.5),         
        ("jules-other", "is:pr author:google-labs-jules[bot] -language:typescript -language:python", 0.3),  
        
        # Low volume agents
        ("devin", "is:pr author:devin-ai-integration[bot]", 2),                   
        ("codegen", "is:pr author:codegen-sh[bot]", 10),                          
        ("claude-code", 'is:pr "Co-Authored-By: Claude"', 1),                    
    ]

    # Shared state across agents
    repos_with_stars: Set[str] = set()
    repo_star_cache: Dict[str, int] = {}
    repo_ai_pr_count: Dict[str, int] = {}

    # Pre-load star counts from previous runs to avoid redundant API calls
    repos_json_path = HUMAN_PR_DIR / "repos_with_500_stars.json"
    existing_repos_json = load_repos_json(repos_json_path)
    for entry in existing_repos_json.get("repos", []):
        repo_star_cache[entry["name"]] = entry.get("stars", 0)
        # repos_with_stars.add(entry["name"])

    # Create directories
    for d in (AI_PR_DIR, HUMAN_REVIEW_DIR, HUMAN_PR_DIR):
        os.makedirs(d, exist_ok=True)

    # ==================================================================
    # Phase 1 - AI-authored PRs + human reviews (append mode)
    # ==================================================================
    print("=" * 80)
    print("Phase 1: Collecting AI-authored PRs and human reviews")
    print("=" * 80 + "\n")

    phase1_stats: Dict[str, dict] = {}

    for agent_name, search_query, chunk_days in agents:
        pr_path = AI_PR_DIR / f"ai_authored_{agent_name}.jsonl"
        review_path = HUMAN_REVIEW_DIR / f"human_reviews_{agent_name}.jsonl"

        existing_pr_ids = load_existing_ids(pr_path)
        existing_review_ids = load_existing_ids(review_path)
        print(f"[{agent_name}] {len(existing_pr_ids)} existing PRs, {len(existing_review_ids)} existing reviews on disk")

        with (
            open(pr_path, "a", encoding="utf-8") as pr_file,
            open(review_path, "a", encoding="utf-8") as review_file,
        ):
            agent_stats = collect_ai_authored_prs(
                client=client,
                pr_file=pr_file,
                review_file=review_file,
                agent_name=agent_name,
                search_query=search_query,
                chunk_days=chunk_days,
                date_range=(from_date, to_date),
                repos_with_stars=repos_with_stars,
                repo_star_cache=repo_star_cache,
                repo_ai_pr_count=repo_ai_pr_count,
                existing_pr_ids=existing_pr_ids,
                existing_review_ids=existing_review_ids,
            )
            phase1_stats[agent_name] = agent_stats

    # Save / merge repos JSON
    merged_json = merge_repos_json(
        existing_repos_json,
        repos_with_stars,
        repo_star_cache,
        repo_ai_pr_count,
        (from_date, to_date),
    )
    with open(repos_json_path, "w", encoding="utf-8") as fh:
        json.dump(merged_json, fh, indent=2)

    print(f"\nPhase 1 complete - {merged_json['count']} total repos tracked in {repos_json_path.name}")
    if repos_with_stars:
        print(
            f"  Star range this run: "
            f"{min(repo_star_cache[r] for r in repos_with_stars):,} - "
            f"{max(repo_star_cache[r] for r in repos_with_stars):,}"
        )

    # ==================================================================
    # Phase 2 - Human-authored PRs (weekly-capped, append mode)
    # ==================================================================
    # Use only repos discovered in this run's Phase 1
    all_repos = repos_with_stars

    if all_repos:
        print("\n" + "=" * 80)
        print("Phase 2: Collecting human-authored PRs (weekly-capped)")
        print("=" * 80 + "\n")

        human_pr_path = HUMAN_PR_DIR / "human_authored_prs.jsonl"
        existing_human_ids = load_existing_ids(human_pr_path)
        print(f"[human] {len(existing_human_ids)} existing PRs on disk")

        with open(human_pr_path, "a", encoding="utf-8") as pr_file:
            phase2_stats = collect_human_authored_prs(
                client=client,
                pr_file=pr_file,
                repos_with_stars=all_repos,
                date_range=(from_date, to_date),
                existing_pr_ids=existing_human_ids,
                weekly_cap=HUMAN_PR_WEEKLY_CAP,
            )
    else:
        print("\nSkipping Phase 2 - no repos discovered.")
        phase2_stats = {}

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    total_ai = sum(s["pr_saved"] for s in phase1_stats.values())
    total_rev = sum(s["human_review_total"] for s in phase1_stats.values())

    print(f"\nPhase 1 (AI-authored PRs):")
    for agent, st in phase1_stats.items():
        print(f"  {agent:25s}: {st['pr_saved']:6,} PRs, {st['human_review_total']:6,} reviews")
    print(f"  {'TOTAL':25s}: {total_ai:6,} PRs, {total_rev:6,} reviews")

    if phase2_stats:
        print(f"\nPhase 2 (Human-authored PRs):")
        print(f"  PRs saved:       {phase2_stats['pr_saved']:,}")
        print(f"  Weeks processed: {phase2_stats['weeks_processed']}")
        print(f"  Weeks capped:    {phase2_stats['weeks_capped']}")

    print(f"\nData directories:")
    print(f"  AI PRs:        {AI_PR_DIR}")
    print(f"  Human reviews: {HUMAN_REVIEW_DIR}")
    print(f"  Human PRs:     {HUMAN_PR_DIR}")
    print("=" * 80)

    client.close()