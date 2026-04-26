"""
Generate PR Arena cumulative PR volume chart.

Expects:
    data/pr_arena/chart-data.json (copy from PR Arena repo docs/chart-data.json)

Outputs:
    figures/pr_arena_growth.pdf
    figures/pr_arena_growth.png
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import DATA_DIR, PROJECT_ROOT

INPUT_FILE = DATA_DIR / "pr_arena" / "chart-data.json"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = FIG_DIR / "pr_arena_growth"

# Agent display config: (json_label_prefix, display_name, color)
AGENT_CONFIG = [
    ("Codex", "OpenAI Codex", "#dc2626"),
    ("Copilot", "GitHub Copilot", "#2563eb"),
    ("Cursor", "Cursor", "#7c3aed"),
    ("Jules", "Google Labs Jules", "#0ea5e9"),
    ("Devin", "Devin", "#059669"),
    ("Codegen", "Codegen", "#d97706"),
]


# ---------------------------------------------------------------------------
# Load and parse
# ---------------------------------------------------------------------------
def load_chart_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_label_date(label: str) -> datetime:
    """Parse PR Arena date labels like '05/26 14:21' or '01/08 15:04'.

    Infers year: months 05-12 -> 2025, months 01-04 -> 2026.
    """
    parts = label.strip().split()
    date_part = parts[0]  # MM/DD
    month = int(date_part.split("/")[0])
    year = 2025 if month >= 5 else 2026
    return datetime.strptime(f"{year}/{label}", "%Y/%m/%d %H:%M")


def extract_agent_totals(data: dict) -> dict:
    """Extract cumulative total PR series for each agent."""
    labels = data["labels"]
    dates = [parse_label_date(l) for l in labels]

    agents = {}
    for dataset in data["datasets"]:
        label = dataset["label"]
        # We want "[Agent] Total" datasets only
        if not label.endswith(" Total"):
            continue

        prefix = label.replace(" Total", "")
        values = dataset["data"]

        # Build (date, value) pairs, skipping nulls
        series = [(d, v) for d, v in zip(dates, values) if v is not None]

        if series:
            agents[prefix] = series

    return agents


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_pr_arena_growth(agents: dict, out_fp: Path) -> None:
    """Grouped bar chart of cumulative PR volume, one bar per agent per date."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )

    # Build color map from AGENT_CONFIG
    color_map = {prefix: color for prefix, _, color in AGENT_CONFIG}
    name_map = {prefix: name for prefix, name, _ in AGENT_CONFIG}

    # Get all dates and agent order
    all_dates = set()
    for series in agents.values():
        for d, _ in series:
            all_dates.add(d)
    dates_sorted = sorted(all_dates)

    # Agent order: largest to smallest (by final value) for visual consistency
    agent_order = sorted(
        agents.keys(),
        key=lambda a: agents[a][-1][1] if agents[a] else 0,
        reverse=True,
    )

    # Build lookup: agent -> date -> value
    lookup = {}
    for agent_key, series in agents.items():
        lookup[agent_key] = {d: v for d, v in series}

    fig, ax = plt.subplots(figsize=(12, 5.5))

    n_agents = len(agent_order)
    n_dates = len(dates_sorted)
    bar_width = 0.7 / n_agents  # Total group width ~0.7

    x = np.arange(n_dates)

    for i, agent_key in enumerate(agent_order):
        values = [lookup.get(agent_key, {}).get(d, 0) for d in dates_sorted]
        offset = (i - n_agents / 2 + 0.5) * bar_width

        ax.bar(
            x + offset,
            values,
            bar_width,
            label=name_map.get(agent_key, agent_key),
            color=color_map.get(agent_key, "#888888"),
            edgecolor="white",
            linewidth=0.3,
        )

    # X-axis labels
    date_labels = [d.strftime("%b '%y") for d in dates_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, rotation=30, ha="right")

    # Y-axis formatting
    def fmt_y(x, _):
        if x == 0:
            return "0"
        if x >= 1e6:
            val = x / 1e6
            return f"{val:.0f}M" if val == int(val) else f"{val:.1f}M"
        val = x / 1e3
        return f"{val:.0f}K" if val == int(val) else f"{val:.1f}K"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))

    ax.set_ylabel("Cumulative PRs")
    ax.set_xlabel("")

    # Clean up
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend with bordered translucent background
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        edgecolor="#cccccc",
        fontsize=10,
    )

    fig.tight_layout()

    # Save
    fig.savefig(out_fp.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_fp.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {out_fp.with_suffix('.pdf').relative_to(PROJECT_ROOT)}")
    print(f"Saved: {out_fp.with_suffix('.png').relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE.relative_to(PROJECT_ROOT)} not found")
        print(f"Copy chart-data.json from PR Arena repo to {INPUT_FILE.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    data = load_chart_data(INPUT_FILE)
    agents = extract_agent_totals(data)

    print("Agents found:")
    for prefix, series in agents.items():
        latest = series[-1]
        print(f"  {prefix:<10}: {len(series)} points, latest = {latest[1]:,.0f} ({latest[0].strftime('%Y-%m-%d')})")

    plot_pr_arena_growth(agents, OUTPUT_FILE)


if __name__ == "__main__":
    main()
