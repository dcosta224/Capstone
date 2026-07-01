#!/usr/bin/env python3
"""Charts for FoodOn contains_* cache and LLM gap-audit results.

Usage:
  uv run python scripts/build_foodon_contains_cache.py
  uv run python scripts/visualize_foodon_contains_results.py
  uv run python scripts/visualize_foodon_contains_results.py --show --out-dir scratch/tag/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from foodon_contains_report_data import load_report

DEFAULT_OUT = ROOT / "scratch" / "tag" / "figures"

PALETTE = {
    "tagged": "#2ecc71",
    "untagged": "#ecf0f1",
    "ancestor": "#3498db",
    "keyword": "#9b59b6",
    "llm_raw": "#f39c12",
    "llm_confirmed": "#27ae60",
    "llm_rejected": "#e74c3c",
    "llm_none": "#95a5a6",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_coverage(report, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [report.tagged_nodes, report.untagged_nodes]
    labels = [
        f"Tagged ({report.tagged_nodes:,})",
        f"Untagged ({report.untagged_nodes:,})",
    ]
    colors = [PALETTE["tagged"], PALETTE["untagged"]]
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(9)
    pct = 100 * report.tagged_nodes / max(report.total_nodes, 1)
    ax.set_title(
        f"FoodOn classes with any contains_* flag\n{report.total_nodes:,} total · {pct:.1f}% tagged",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, out_dir, "01_coverage")


def chart_cache_by_slug(report, out_dir: Path) -> Path:
    slugs = sorted(report.cache_counts, key=lambda s: report.cache_counts[s], reverse=True)
    counts = [report.cache_counts[s] for s in slugs]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(slugs))
    bars = ax.barh(y, counts, color=PALETTE["tagged"], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(slugs)
    ax.invert_yaxis()
    ax.set_xlabel("FoodOn classes tagged")
    ax.set_title("Production cache: tagged classes per dimension", fontweight="bold")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2, f"{n:,}", va="center", fontsize=8)
    return _save(fig, out_dir, "02_cache_by_slug")


def chart_build_methods(report, out_dir: Path) -> Path:
    slugs = sorted(report.cache_counts, key=lambda s: report.cache_counts[s], reverse=True)[:10]
    ancestor = [report.ancestor_only_counts.get(s, 0) for s in slugs]
    keyword = [report.label_keyword_additions.get(s, 0) for s in slugs]
    x = np.arange(len(slugs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ancestor, width, label="Ancestor propagation only", color=PALETTE["ancestor"])
    ax.bar(x + width / 2, keyword, width, label="Extra from label keywords", color=PALETTE["keyword"])
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, rotation=35, ha="right")
    ax.set_ylabel("Class count")
    ax.set_title("How tags enter the cache (top 10 dimensions)", fontweight="bold")
    ax.legend()
    return _save(fig, out_dir, "03_build_methods")


def chart_llm_funnel(report, out_dir: Path) -> Path:
    if report.llm_review_rows == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No LLM review CSV found", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out_dir, "04_llm_funnel")

    stages = [
        ("Untagged food-like\ncandidates", report.llm_review_rows),
        ("LLM suggested\ntags", report.llm_raw_suggestions),
        ("Ontology\nconfirmed", report.llm_confirmed),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [PALETTE["llm_none"], PALETTE["llm_raw"], PALETTE["llm_confirmed"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Class count")
    ax.set_title("LLM gap audit funnel (with ontology guardrails)", fontweight="bold")
    for bar, n, i in zip(bars, values, x):
        ax.text(bar.get_x() + bar.get_width() / 2, n + max(values) * 0.02, f"{n:,}", ha="center", fontsize=10)
        if i > 0 and values[i - 1]:
            drop = 100 * (1 - n / values[i - 1])
            ax.text(bar.get_x() + bar.get_width() / 2, n / 2, f"-{drop:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    rejected = report.llm_raw_suggestions - report.llm_confirmed
    ax.text(
        0.98,
        0.95,
        f"Rejected by ontology: {rejected:,}\nNo LLM tag: {report.llm_no_suggestion:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#fef9e7", alpha=0.9),
    )
    return _save(fig, out_dir, "04_llm_funnel")


def chart_llm_confirmed_slugs(report, out_dir: Path) -> Path:
    if not report.confirmed_slug_counts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No ontology-confirmed LLM tags", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out_dir, "05_llm_confirmed_slugs")

    slugs = sorted(report.confirmed_slug_counts, key=report.confirmed_slug_counts.get, reverse=True)
    counts = [report.confirmed_slug_counts[s] for s in slugs]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(slugs))
    ax.barh(y, counts, color=PALETTE["llm_confirmed"])
    ax.set_yticks(y)
    ax.set_yticklabels(slugs)
    ax.invert_yaxis()
    ax.set_xlabel("Ontology-confirmed classes")
    ax.set_title("LLM audit: confirmed additions by dimension", fontweight="bold")
    for i, n in enumerate(counts):
        ax.text(n + max(counts) * 0.02, i, str(n), va="center", fontsize=9)
    return _save(fig, out_dir, "05_llm_confirmed_slugs")


def chart_pipeline_schematic(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, "FoodOn\nontology", "#d5e8f7"),
        (2.0, 1.2, "Rule build\n(ancestors +\nkeywords)", "#d5f5e3"),
        (4.2, 1.2, "foodon_\ncontains.csv", "#fdebd0"),
        (6.2, 1.2, "USDA →\nFoodOn map", "#e8daef"),
        (8.0, 1.2, "User tags\ndairy_free…", "#fadbd8"),
    ]
    for i, (x, y, text, color) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x, y), 1.5, 1.2, facecolor=color, edgecolor="#566573", linewidth=1.2))
        ax.text(x + 0.75, y + 0.6, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0] - 0.05, y + 0.6), xytext=(x + 1.55, y + 0.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(2.0, 0.35, "Gap audit: LLM → ontology validate → review queue", ha="center", fontsize=9, style="italic", color="#566573")
    ax.set_title("End-to-end tagging pipeline", fontweight="bold", pad=8)
    return _save(fig, out_dir, "00_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FoodOn contains tagging results")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show", action="store_true", help="Open figures after saving")
    args = parser.parse_args()

    report = load_report()
    paths = [
        chart_pipeline_schematic(args.out_dir),
        chart_coverage(report, args.out_dir),
        chart_cache_by_slug(report, args.out_dir),
        chart_build_methods(report, args.out_dir),
        chart_llm_funnel(report, args.out_dir),
        chart_llm_confirmed_slugs(report, args.out_dir),
    ]

    print(f"Wrote {len(paths)} figures to {args.out_dir.resolve()}", flush=True)
    for p in paths:
        print(f"  {p.name}", flush=True)

    if args.show:
        import subprocess

        for p in paths:
            subprocess.run(["start", "", str(p)], shell=True, check=False)


if __name__ == "__main__":
    main()
