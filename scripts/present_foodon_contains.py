#!/usr/bin/env python3
"""Speaker script for a short presentation on FoodOn contains_* tagging.

Prints timed talking points and optionally generates/refers to chart PNGs.

Usage:
  uv run python scripts/present_foodon_contains.py
  uv run python scripts/present_foodon_contains.py --minutes 5 --write-notes scratch/tag/presentation_notes.txt
  uv run python scripts/present_foodon_contains.py --figures
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from foodon_contains_report_data import load_report

FIGURE_DIR = ROOT / "scratch" / "tag" / "figures"


def _wrap(text: str, width: int = 78) -> str:
    return "\n".join(textwrap.fill(line, width=width) if line.strip() else "" for line in text.splitlines())


def _section(title: str, minutes: float, body: str) -> str:
    bar = "-" * 72
    return f"\n{bar}\n[{minutes:.0f} min] {title}\n{bar}\n\n{_wrap(body)}\n"


def build_script(report, total_minutes: float) -> str:
    pct_tagged = 100 * report.tagged_nodes / max(report.total_nodes, 1)
    top_slug = max(report.cache_counts, key=report.cache_counts.get)
    filter_pct = (
        100 * (1 - report.llm_confirmed / report.llm_raw_suggestions)
        if report.llm_raw_suggestions
        else 0
    )

    samples = ""
    for s in report.sample_confirmed[:3]:
        samples += f'\n  - "{s["label"]}" -> +{s["confirmed"]}'

    # Scale section times to requested length (default ~4 min content)
    m1, m2, m3, m4 = 0.75, 1.0, 1.0, 1.25
    scale = total_minutes / (m1 + m2 + m3 + m4)

    parts = [
        _section(
            f"1. Problem & goal",
            m1 * scale,
            f"""We need to know whether a food contains dairy, fish, wheat, and similar ingredients
so we can assign user-facing tags like dairy_free or vegan.

FoodOn is a large food ontology — {report.total_nodes:,} classes — but users and USDA foods
connect through a mapped FoodOn ID. We precompute low-level contains_* flags on FoodOn
classes, then derive restriction tags when an ingredient is linked.""",
        ),
        _section(
            f"2. Rule-based tagging (primary method)",
            m2 * scale,
            f"""We do NOT ask an LLM to tag every class. The production cache is built from
data/diet_tags.json rules:

  final = (ancestor propagation + label keywords) - excludes - manual overrides

- Ancestor propagation -- e.g. everything under "fish food product" gets contains_fish.
- Label keywords -- catch classes whose label says "chicken stew" even if they sit off the
  main poultry branch.
- Excludes -- e.g. peanut subtree removed from tree_nut (peanut is a legume).
- Suppress phrases -- e.g. "peanut butter" must not trigger dairy via the word butter.

Result today: {report.tagged_nodes:,} of {report.total_nodes:,} classes tagged ({pct_tagged:.1f}%).
The largest dimension in cache is {top_slug} ({report.cache_counts[top_slug]:,} classes).
About two-thirds stay untagged on purpose — many classes are processes, codes, or plant
foods outside our 15 restriction dimensions.""",
        ),
        _section(
            f"3. LLM gap audit (secondary, controlled)",
            m3 * scale,
            f"""Rules miss edge cases — especially "piece of catfish meat" style classes on
anatomy branches, not under the main fish product subtree.

We audited {report.llm_review_rows:,} untagged, food-like candidates with a local Qwen model.
The LLM proposed tags for {report.llm_raw_suggestions:,} classes — but suggestions are NOT
applied automatically.

An ontology validator checks each proposal against:
  - FoodOn parent/sibling/descendant tags
  - keyword agreement in diet_tags.json
  - taxonomy penalties (eurocode / category nodes)
  - high-confidence "piece of ... meat" fast-path for animal products

Only {report.llm_confirmed:,} passed ({filter_pct:.0f}% of LLM suggestions filtered out).
These live in a human review queue — not yet merged into the production cache.""",
        ),
        _section(
            f"4. Results & takeaway",
            m4 * scale,
            f"""Production cache (rule-based):
  - {report.tagged_nodes:,} tagged / {report.untagged_nodes:,} untagged FoodOn classes
  - 15 contains dimensions (dairy, fish, wheat, poultry, ...)

LLM audit (review queue):
  - {report.llm_review_rows:,} candidates -> {report.llm_raw_suggestions:,} LLM suggestions
    -> {report.llm_confirmed:,} ontology-confirmed
  - Top confirmed gaps: red_meat, poultry, fish, wheat

Examples confirmed by ontology:{samples or chr(10) + "  (run audit to populate)"}

Takeaway: ontology-first, explainable tagging with LLM as a guarded gap-finder -- not an
arbitrary classifier. Next step: promote reviewed rows into overrides or new ancestor roots,
rebuild cache, then flow contains_* -> dairy_free / vegan on linked USDA ingredients.""",
        ),
    ]

    figure_block = f"""
{'=' * 72}
VISUAL AIDS  (scratch/tag/figures/)
{'=' * 72}
  00_pipeline.png          - end-to-end flow (show first)
  01_coverage.png          - tagged vs untagged pie chart
  02_cache_by_slug.png     - production counts per dimension
  03_build_methods.png     - ancestors vs label keywords
  04_llm_funnel.png        - audit funnel with filter rate
  05_llm_confirmed_slugs.png - confirmed gap breakdown

Generate charts: uv run python scripts/visualize_foodon_contains_results.py
"""
    return "".join(parts) + figure_block


def main() -> None:
    parser = argparse.ArgumentParser(description="Print speaker script for FoodOn contains tagging")
    parser.add_argument("--minutes", type=float, default=4.0, help="Target presentation length")
    parser.add_argument("--write-notes", type=Path, default=None, help="Save script to this file")
    parser.add_argument("--figures", action="store_true", help="Generate chart PNGs before printing")
    parser.add_argument("--quiet-header", action="store_true")
    args = parser.parse_args()

    if args.figures:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "visualize_foodon_contains_results.py")],
            cwd=ROOT,
            check=True,
        )

    report = load_report()
    script = build_script(report, args.minutes)

    if not args.quiet_header:
        print("=" * 72)
        print("FOODON CONTAINS TAGGING - SPEAKER SCRIPT")
        print(f"Target length: ~{args.minutes:.0f} minutes | figures: {FIGURE_DIR}")
        print("=" * 72)

    print(script)

    if args.write_notes:
        args.write_notes.parent.mkdir(parents=True, exist_ok=True)
        header = f"FoodOn contains_* tagging - speaker notes (~{args.minutes:.0f} min)\n"
        args.write_notes.write_text(header + script, encoding="utf-8")
        print(f"\nWrote {args.write_notes.resolve()}", flush=True)


if __name__ == "__main__":
    main()
