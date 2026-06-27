#!/usr/bin/env python3
"""Validate FoodOn contains cache against probe foods and optional FDC mapping.

Usage:
  uv run python scripts/build_foodon_contains_cache.py
  uv run python scripts/audit_foodon_contains.py
  uv run python scripts/audit_foodon_contains.py --mapping
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from diet_tags_core import load_diet_tags, tag_ingredient
from foodon_contains_core import load_contains_table, lookup_contains
from foodon_index import FoodOnIndex
from foodon_mapping_io import load_mapping_lookup
from foodon_paths import FOODON_INDEX_CACHE

# label -> expected contains slugs (ontology-only path via linked foodon_id)
PROBE_FOODS: dict[str, list[str]] = {
    "cheddar cheese": ["dairy"],
    "cow milk": ["dairy"],
    "chicken broth": ["poultry"],
    "beef broth": ["red_meat"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FoodOn contains cache")
    parser.add_argument("--mapping", action="store_true", help="Also audit mapped FDC rows")
    parser.add_argument("--limit", type=int, default=200, help="Max mapped FDC rows to sample")
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

    print("=== Probe foods (FoodOn label -> cache) ===", flush=True)
    failures = 0
    for label, expected in PROBE_FOODS.items():
        node_id = _resolve_foodon_id(index, label)
        if node_id is None:
            print(f"  FAIL {label!r}: no FoodOn match", flush=True)
            failures += 1
            continue
        got = sorted(lookup_contains(node_id, table))
        expected_sorted = sorted(expected)
        ok = set(expected) <= set(got)
        status = "ok" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  {status} {label!r} [{node_id}]: expected {expected_sorted or ['(none)']}, got {got or ['(none)']}",
            flush=True,
        )

    print("\n=== Peanut vs tree_nut separation ===", flush=True)
    for label in ("peanut", "almond"):
        node_id = index.search(label, limit=1)[0]["id"]
        got = lookup_contains(node_id, table)
        print(f"  {label} [{node_id}]: {sorted(got)}", flush=True)
        if label == "peanut" and "tree_nut" in got:
            failures += 1
            print("    FAIL: peanut should not contain tree_nut", flush=True)
        if label == "almond" and "tree_nut" not in got:
            failures += 1
            print("    FAIL: almond should contain tree_nut", flush=True)

    if args.mapping:
        mapping = load_mapping_lookup()
        if not mapping:
            print("\nNo fdc_foodon_mapping found in scratch/tag/", flush=True)
        else:
            print(f"\n=== Mapped FDC sample (n={min(args.limit, len(mapping))}) ===", flush=True)
            keyword_only = 0
            cache_only = 0
            both = 0
            neither = 0
            for i, (fdc_id, row) in enumerate(mapping.items()):
                if i >= args.limit:
                    break
                desc = str(row.get("description") or row.get("fdc_description") or "")
                node_id = str(row["foodon_id"])
                kw = tag_ingredient(
                    int(fdc_id),
                    desc,
                    None,
                    registry,
                    foodon_index=None,
                    foodon_node_id=None,
                )["contains_set"]
                full = tag_ingredient(
                    int(fdc_id),
                    desc,
                    None,
                    registry,
                    foodon_index=None,
                    foodon_node_id=node_id,
                    foodon_contains_table=table,
                )["contains_set"]
                cache = full - kw
                if kw and cache:
                    both += 1
                elif kw:
                    keyword_only += 1
                elif cache:
                    cache_only += 1
                else:
                    neither += 1
            stats = {
                "sampled": min(args.limit, len(mapping)),
                "keyword_only": keyword_only,
                "cache_only": cache_only,
                "both": both,
                "neither": neither,
            }
            print(json.dumps(stats, indent=2), flush=True)

    print(f"\nProbe failures: {failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
