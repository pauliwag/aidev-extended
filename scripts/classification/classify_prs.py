"""
Classify AI and human PRs into conventional commit types using:
1. Regex matching on PR title
2. LLM classification for unmatched PRs (using title + body)
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import (
    AI_PR_DIR,
    HUMAN_PR_DIR,
    PROCESSED_DIR,
    AGENTS,
)

# Output directory
CLASSIFIED_DIR = PROCESSED_DIR / "classified_prs"
CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency settings
MAX_WORKERS = 10  # Parallel LLM calls
CHECKPOINT_INTERVAL = 20  # Save every N PRs

# Body truncation for LLM prompt (characters)
BODY_PROMPT_CAP = 500

# Limit PRs per agent for quick validation
TEST_MODE = "--test" in sys.argv
SAMPLE_LIMIT = 5

if TEST_MODE:
    print("\n" + "=" * 60)
    print(f"🧪 TEST MODE: Processing max {SAMPLE_LIMIT} PRs per agent")
    print("=" * 60 + "\n")

# -----------------------------------------------------------------------------
# Conventional Commit Types
# -----------------------------------------------------------------------------
TYPES = {
    "feat": "A new feature",
    "fix": "A bug fix",
    "docs": "Documentation only changes",
    "style": "Changes that do not affect the meaning of the code (white-space, formatting, etc)",
    "refactor": "A code change that neither fixes a bug nor adds a feature",
    "perf": "A code change that improves performance",
    "test": "Adding missing tests or correcting existing tests",
    "build": "Changes that affect the build system or external dependencies",
    "ci": "Changes to our CI configuration files and scripts",
    "chore": "Changes to the build process or auxiliary tools",
    "other": "Any other changes that do not fit the above categories",
    "revert": "Reverts a previous commit",
}

# Compile regex patterns for conventional commit detection
PATTERNS = {
    t: re.compile(rf"^{t}(\([^)]*\))?!?(?=\W|$)", flags=re.IGNORECASE)
    for t in TYPES.keys()
}


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_prs_from_jsonl(filepath: Path) -> list[dict]:
    """Load PRs from a JSONL file."""
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist")
        return []

    prs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prs.append(json.loads(line))
    return prs


def truncate_body_for_prompt(body: str | None) -> str:
    """Truncate PR body for inclusion in LLM prompt."""
    if not body or not isinstance(body, str):
        return ""
    body = body.strip()
    if len(body) <= BODY_PROMPT_CAP:
        return body
    return body[:BODY_PROMPT_CAP] + "..."


def title_label(title: str) -> str | None:
    """Stage 1: Regex-based classification from PR title."""
    if not title or not isinstance(title, str):
        return None

    first_line = title.splitlines()[0].strip()
    for typ, pattern in PATTERNS.items():
        if pattern.match(first_line):
            return typ.lower()
    return None


def classify(title: str, body: str = "", max_retries: int = 3) -> tuple[str, str, int]:
    """Stage 2: LLM classification using title and body. Returns (reason, output, confidence)."""
    types_str = "\n".join([f"{k}: {v}" for k, v in TYPES.items()])

    body_section = ""
    if body:
        body_section = f"\nPR Body: {body}\n"

    prompt = f"""You are a Conventional Commit classifier.
Given a PR title{" and body" if body else ""}, pick **exactly one** label from:
{types_str}

PR Title: {title}
{body_section}

Respond with JSON containing:
- reason: A brief explanation for why this commit type was chosen
- output: One of the allowed Conventional Commit types
- confidence: Confidence score (1-10)
"""

    format_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "output": {"type": "string", "enum": list(TYPES.keys())},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["reason", "output", "confidence"],
    }

    for attempt in range(max_retries):
        try:
            response = ollama.generate(
                model="qwen3:30b-instruct",
                prompt=prompt,
                stream=False,
                format=format_schema,
                options=ollama.Options(
                    temperature=0.1,  # Lower temp for more determinism
                ),
            ).response

            data = json.loads(response)

            # Validate output
            if data["output"] not in TYPES:
                print(f"Invalid category: {data['output']}, retrying...")
                if attempt < max_retries - 1:
                    continue
                data["output"] = "other"

            return data["reason"], data["output"], int(data["confidence"])

        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

        except Exception as e:
            print(f"Classification error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue

    # Failed after all retries
    return "Classification failed after retries", "other", 1


# -----------------------------------------------------------------------------
# Classification Logic
# -----------------------------------------------------------------------------
def classify_prs(prs: list[dict], agent: str) -> None:
    """
    Classify a list of PRs for a given agent.
    Writes results to JSONL file.
    """
    out_fp = CLASSIFIED_DIR / f"{agent}_pr_task_type.jsonl"

    # Load existing results if available
    done_ids = set()
    if out_fp.exists():
        with open(out_fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    done_ids.add(str(result["id"]))

    # Filter out already processed PRs
    prs_to_process = [pr for pr in prs if str(pr["id"]) not in done_ids]

    # Apply test mode limit
    if TEST_MODE and len(prs_to_process) > SAMPLE_LIMIT:
        prs_to_process = prs_to_process[:SAMPLE_LIMIT]
        print(f"[{agent}] 🧪 Test mode: limiting to {SAMPLE_LIMIT} PRs")

    if not prs_to_process:
        print(f"[{agent}] All PRs already processed")
        return

    print(f"\n[{agent}] Processing {len(prs_to_process)} PRs...")

    # Stage 1: Regex-based classification
    stage1_results = []
    llm_needed = []

    for pr in prs_to_process:
        title = pr.get("title", "") or ""
        label = title_label(title)

        if label:
            stage1_results.append(
                {
                    "agent": agent,
                    "id": str(pr["id"]),
                    "title": title,
                    "reason": "title provides conventional commit label",
                    "type": label,
                    "confidence": 10,
                }
            )
        else:
            llm_needed.append(pr)

    print(f"[{agent}] Stage 1: {len(stage1_results)} classified via regex")

    # Save Stage 1 results immediately (append to JSONL)
    if stage1_results:
        with open(out_fp, "a", encoding="utf-8") as f:
            for result in stage1_results:
                f.write(json.dumps(result) + "\n")

    if not llm_needed:
        return

    # Stage 2: LLM classification
    print(f"[{agent}] Stage 2: {len(llm_needed)} PRs need LLM classification")

    def process_pr(pr: dict) -> dict | None:
        try:
            title = pr.get("title", "") or ""
            body = truncate_body_for_prompt(pr.get("body"))
            pr_id = str(pr["id"])

            reason, label, confidence = classify(title, body)
            print(f"  [{pr_id}] -> {label}: {reason[:50]}... (conf {confidence})")

            return {
                "agent": agent,
                "id": pr_id,
                "title": title,
                "reason": reason,
                "type": label,
                "confidence": confidence,
            }
        except Exception as e:
            print(f"Error processing PR {pr.get('id')}: {e}")
            return None

    buffer = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pr = {executor.submit(process_pr, pr): pr for pr in llm_needed}

        for i, future in enumerate(as_completed(future_to_pr), 1):
            result = future.result()
            if result:
                buffer.append(result)

            # Checkpoint periodically (append to JSONL)
            if len(buffer) >= CHECKPOINT_INTERVAL:
                with open(out_fp, "a", encoding="utf-8") as f:
                    for res in buffer:
                        f.write(json.dumps(res) + "\n")
                print(f"[{agent}] Checkpoint: saved {len(buffer)} PRs ({i}/{len(llm_needed)} processed)")
                buffer = []

    # Final save
    if buffer:
        with open(out_fp, "a", encoding="utf-8") as f:
            for res in buffer:
                f.write(json.dumps(res) + "\n")
        print(f"[{agent}] Final save: {len(buffer)} PRs")

    print(f"[{agent}] ✓ Complete")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PR Classification Script")
    print("=" * 60)

    # Process AI-authored PRs
    for agent in AGENTS:
        pr_file = AI_PR_DIR / f"ai_authored_{agent}.jsonl"
        if not pr_file.exists():
            print(f"\nSkipping {agent}: file not found")
            continue

        prs = load_prs_from_jsonl(pr_file)
        if not prs:
            print(f"\nNo PRs found for {agent}")
            continue

        classify_prs(prs, agent)

    # Process human-authored PRs
    human_pr_file = HUMAN_PR_DIR / "human_authored_prs.jsonl"
    if human_pr_file.exists():
        human_prs = load_prs_from_jsonl(human_pr_file)
        if human_prs:
            classify_prs(human_prs, "Human")

    print("\n" + "=" * 60)
    print("Classification complete!")
    print(f"Results saved to: {CLASSIFIED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
