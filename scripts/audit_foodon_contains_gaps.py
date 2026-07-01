#!/usr/bin/env python3
"""Audit FoodOn contains cache: probes, label/ancestor gaps, and disagreement stats.

Usage:
  uv run python scripts/build_foodon_contains_cache.py
  uv run python scripts/audit_foodon_contains_gaps.py
  uv run python scripts/audit_foodon_contains_gaps.py --export scratch/tag/foodon_contains_gaps.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from diet_tags_core import contains_slugs_from_label, load_diet_tags
from foodon_contains_core import build_contains_lookup, load_contains_table, lookup_contains
from foodon_index import FoodOnIndex
from foodon_paths import FOODON_CONTAINS_SUMMARY, FOODON_INDEX_CACHE

# label -> minimum expected contains slugs (after hybrid ancestor + label tagging)
PROBE_FOODS: dict[str, list[str]] = {
    "cheddar cheese": ["dairy"],
    "cow milk": ["dairy"],
    "chicken broth": ["poultry"],
    "chicken stew or hash": ["poultry"],
    "poultry stew": ["poultry"],
    "beef broth": ["red_meat"],
    "beef stew": ["red_meat"],
    "chicken breast": ["poultry"],
    "almond": ["tree_nut"],
    "peanut": ["peanut"],
    "peanut butter": ["peanut"],
    "walnut": ["tree_nut"],
    "onion": ["root_vegetable"],
    "potato": ["root_vegetable"],
    "shrimp food product": ["shellfish"],
    "salmon": ["fish"],
    "tofu": ["soy"],
    "wheat flour": ["wheat"],
    "honey food product": ["honey"],
    "olive oil": [],
}


def _resolve_foodon_id(index: FoodOnIndex, label: str) -> str | None:
    label_to_id = {v.lower(): k for k, v in index.labels.items()}
    exact = label_to_id.get(label.lower())
    if exact is not None:
        return exact
    hits = index.search(label, limit=12)
    foodon_hits = [h for h in hits if str(h["id"]).startswith("FOODON_")]
    if foodon_hits:
        return str(foodon_hits[0]["id"])
    return str(hits[0]["id"]) if hits else None


def _probe_failures(index: FoodOnIndex, table: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    print("=== Probe foods ===", flush=True)
    for label, expected in PROBE_FOODS.items():
        node_id = _resolve_foodon_id(index, label)
        if node_id is None:
            msg = f"{label!r}: no FoodOn match"
            failures.append(msg)
            print(f"  FAIL {msg}", flush=True)
            continue
        got = sorted(lookup_contains(node_id, table))
        ok = set(expected) <= set(got)
        status = "ok" if ok else "FAIL"
        print(
            f"  {status} {label!r} [{node_id}]: expected {sorted(expected) or ['(none)']}, got {got or ['(none)']}",
            flush=True,
        )
        if not ok:
            failures.append(f"{label!r} [{node_id}]: expected {expected}, got {got}")
    return failures


def _label_gaps(
    index: FoodOnIndex,
    table: pd.DataFrame,
    registry,
    *,
    export_path: Path | None,
) -> list[str]:
    """Rows where label keywords expect a tag the cache does not have."""
    failures: list[str] = []
    rows: list[dict] = []
    cols = [c for c in table.columns if c.startswith("contains_")]

    for _, row in table.iterrows():
        node_id = str(row["foodon_id"])
        label = str(row["label"])
        expected = contains_slugs_from_label(label, registry)
        actual = lookup_contains(node_id, table)
        missing = expected - actual
        if not missing:
            continue
        failures.append(f"{label!r} [{node_id}]: label implies {sorted(expected)}, missing {sorted(missing)}")
        rows.append(
            {
                "foodon_id": node_id,
                "label": label,
                "expected_from_label": ",".join(sorted(expected)),
                "actual": ",".join(sorted(actual)),
                "missing": ",".join(sorted(missing)),
            }
        )

    print(f"\n=== Label keyword gaps (false negatives) ===", flush=True)
    print(f"  count: {len(rows)}", flush=True)
    for line in failures[:15]:
        print(f"  {line}", flush=True)
    if len(failures) > 15:
        print(f"  ... and {len(failures) - 15} more", flush=True)

    if export_path and rows:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(export_path, index=False)
        print(f"  Wrote {export_path}", flush=True)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FoodOn contains gaps")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write label-gap rows to CSV",
    )
    args = parser.parse_args()

    if not FOODON_INDEX_CACHE.is_file():
        print("Run: uv run python scripts/build_foodon_index_cache.py", flush=True)
        raise SystemExit(1)

    table = load_contains_table()
    if table is None:
        print("Run: uv run python scripts/build_foodon_contains_cache.py", flush=True)
        raise SystemExit(1)

    index = FoodOnIndex.from_cache(FOODON_INDEX_CACHE)
    registry = load_diet_tags()

    failures = _probe_failures(index, table)
    failures.extend(_label_gaps(index, table, registry, export_path=args.export))

    if FOODON_CONTAINS_SUMMARY.is_file():
        summary = json.loads(FOODON_CONTAINS_SUMMARY.read_text(encoding="utf-8"))
        print("\n=== Build summary (label keyword additions) ===", flush=True)
        additions = summary.get("label_keyword_additions") or {}
        for slug, count in sorted(additions.items(), key=lambda x: -x[1]):
            if count:
                print(f"  {slug}: +{count} classes from label keywords", flush=True)

    anc_df, _ = build_contains_lookup(index, use_label_keywords=False)
    full_df, _ = build_contains_lookup(index, use_label_keywords=True)
    cols = [c for c in full_df.columns if c.startswith("contains_")]
    label_only = 0
    for slug_col in cols:
        anc_on = anc_df[slug_col].astype(bool)
        full_on = full_df[slug_col].astype(bool)
        label_only += int((full_on & ~anc_on).sum())
    print(f"\n=== Hybrid tagging ===", flush=True)
    print(f"  classes tagged via label keywords only (not ancestor): {label_only}", flush=True)

    print(f"\nTotal failures: {len(failures)}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
