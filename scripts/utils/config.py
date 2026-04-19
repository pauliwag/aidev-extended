"""Configuration and paths for the project."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data subdirectories
AI_PR_DIR = RAW_DIR / "ai_authored_prs"
HUMAN_REVIEW_DIR = RAW_DIR / "human_reviews"
HUMAN_PR_DIR = RAW_DIR / "human_authored_prs"

# Processed data subdirectories
PROCESSED_AI_PR_DIR = PROCESSED_DIR / "ai_authored_prs"
PROCESSED_HUMAN_REVIEW_DIR = PROCESSED_DIR / "human_reviews"

# Configuration
MIN_STARS = int(os.getenv("MIN_STARS", 500))
AGENTS = [
    "copilot",
    "cursor",
    "jules",
    "devin",
    # "codegen",
    "claude-code",
    "codex",
]
