#!/usr/bin/env python3
"""Compare keyword vs FoodOn-cache tagging on mapped ingredients.

Usage:
  uv run python scripts/tag_ingredients.py --mapping --mapped-only
  uv run python scripts/analyze_ingredient_tags.py
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

from diet_tags_core import load_diet_tags, tag_ingredient
from diet_tags_io import load_foods_catalog
from foodon_contains_core import load_contains_table
from foodon_mapping_io import load_mapping, load_mapping_lookup

OUT_DIR = ROOT / "scratch" / "tag"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mapped ingredient diet tags")
    parser.add_argument("--low-confidence", type=float, default=0.75)
    args = parser.parse_args()

    mapping_df = load_mapping()
    if mapping_df.empty:
        print("No mapping table found.", flush=True)
        raise SystemExit(1)

    contains_table = load_contains_table()
    if contains_table is None:
        print("Run: uv run python scripts/build_foodon_contains_cache.py", flush=True)
        raise SystemExit(1)

    registry = load_diet_tags()
    lookup = load_mapping_lookup()
    foods = load_foods_catalog(fdc_ids=set(lookup.keys()))
    food_by_id = {int(r.fdc_id): r for r in foods.itertuples(index=False)}

    rows: list[dict] = []
    stats = {
        "mapped_total": len(lookup),
        "foods_in_catalog": len(food_by_id),
        "keyword_only": 0,
        "cache_only": 0,
        "both": 0,
        "neither": 0,
        "low_confidence_with_contains": 0,
    }

    for fdc_id, meta in lookup.items():
        food = food_by_id.get(fdc_id)
        if food is None:
            continue
        desc = str(food.description)
        ing = (
            str(food.ingredients)
            if food.ingredients is not None and pd.notna(food.ingredients)
            else None
        )
        node_id = meta["foodon_id"]

        kw = tag_ingredient(
            fdc_id, desc, ing, registry, nutrient_lookup={}, foodon_index=None
        )["contains_set"]
        full = tag_ingredient(
            fdc_id,
            desc,
            ing,
            registry,
            nutrient_lookup={},
            foodon_node_id=node_id,
            foodon_contains_table=contains_table,
        )["contains_set"]
        cache_only = full - kw
        kw_only = kw - full

        if kw and cache_only:
            stats["both"] += 1
            bucket = "both_plus_cache"
        elif kw and not cache_only:
            stats["keyword_only"] += 1
            bucket = "keyword_only"
        elif cache_only:
            stats["cache_only"] += 1
            bucket = "cache_only"
        else:
            stats["neither"] += 1
            bucket = "neither"

        conf = float(meta.get("confidence") or 0.0)
        if full and conf < args.low_confidence:
            stats["low_confidence_with_contains"] += 1

        if cache_only or kw_only or (conf < args.low_confidence and full):
            rows.append(
                {
                    "fdc_id": fdc_id,
                    "description": desc,
                    "foodon_id": node_id,
                    "foodon_label": meta.get("foodon_label", ""),
                    "match_method": meta.get("match_method", ""),
                    "confidence": conf,
                    "bucket": bucket,
                    "keyword_contains": ",".join(sorted(kw)),
                    "cache_contains": ",".join(sorted(full)),
                    "cache_only_adds": ",".join(sorted(cache_only)),
                    "keyword_only_adds": ",".join(sorted(kw_only)),
                }
            )

    review_df = pd.DataFrame(rows).sort_values(
        ["confidence", "fdc_id"], ascending=[True, True]
    )
    review_path = OUT_DIR / "ingredient_tag_review.csv"
    review_df.to_csv(review_path, index=False)

    summary_path = OUT_DIR / "ingredient_tag_analysis.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2), flush=True)
    print(f"Wrote {review_path} ({len(review_df):,} noteworthy rows)", flush=True)
    print(f"Wrote {summary_path}", flush=True)

    if not review_df.empty:
        print("\nSample cache-only additions:", flush=True)
        cache_rows = review_df[review_df["cache_only_adds"].astype(str).str.len() > 0].head(8)
        for row in cache_rows.itertuples(index=False):
            print(
                f"  [{row.confidence:.2f}] {row.description!r} -> {row.foodon_label!r}: +{row.cache_only_adds}",
                flush=True,
            )


if __name__ == "__main__":
    main()
