#!/usr/bin/env python3
"""Precompute FoodOn class → contains flags from ontology ancestor roots.

Reads:
  - scratch/foodon_index.json
  - data/diet_tags.json (contains.foodon_ancestors)
  - data/foodon_contains_overrides.json (optional)

Writes:
  - scratch/tag/foodon_contains.parquet
  - scratch/tag/foodon_contains_summary.json

Usage:
  uv run python scripts/build_foodon_contains_cache.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_contains_core import build_contains_lookup, write_contains_table
from foodon_index import FoodOnIndex
from foodon_paths import FOODON_CONTAINS_SUMMARY, FOODON_INDEX_CACHE

SAMPLE_LABELS = (
    "cheddar cheese",
    "chicken broth",
    "beef broth",
    "wheat flour",
    "peanut butter",
    "shrimp food product",
    "tofu",
    "honey food product",
    "olive oil",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FoodOn contains cache")
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--all-nodes", action="store_true", help="Include non-FOODON_* classes")
    args = parser.parse_args()

    if not FOODON_INDEX_CACHE.is_file():
        print("Run: uv run python scripts/build_foodon_index_cache.py", flush=True)
        raise SystemExit(1)

    t0 = time.perf_counter()
    index = FoodOnIndex.from_cache(FOODON_INDEX_CACHE)
    df, summary = build_contains_lookup(
        index,
        overrides_path=args.overrides,
        foodon_only=not args.all_nodes,
    )
    out = write_contains_table(df, summary)
    elapsed = time.perf_counter() - t0

    print(f"FoodOn classes indexed: {len(index.labels):,}", flush=True)
    print(f"Contains table rows: {len(df):,}", flush=True)
    print(f"Wrote {out} ({elapsed:.1f}s)", flush=True)
    print(f"Summary: {FOODON_CONTAINS_SUMMARY}", flush=True)
    print("\nTagged class counts per contains slug:", flush=True)
    for slug, n in sorted(summary["tagged_counts"].items(), key=lambda x: -x[1]):
        print(f"  {slug}: {n:,}", flush=True)

    print("\nSample lookups:", flush=True)
    label_to_id = {v.lower(): k for k, v in index.labels.items()}
    contains_cols = [c for c in df.columns if c.startswith("contains_")]
    for q in SAMPLE_LABELS:
        node_id = label_to_id.get(q.lower())
        if node_id is None:
            hits = index.search(q, limit=1)
            if not hits:
                print(f"  {q!r}: (no match)", flush=True)
                continue
            node_id = hits[0]["id"]
        row = df[df["foodon_id"] == node_id]
        if row.empty:
            print(f"  {q!r} [{node_id}]: (not in FOODON table)", flush=True)
            continue
        r = row.iloc[0]
        flags = [c.replace("contains_", "") for c in contains_cols if bool(r[c])]
        print(f"  {r['label']!r} [{node_id}]: {flags or ['(none)']}", flush=True)


if __name__ == "__main__":
    main()
