#!/usr/bin/env python3
"""Tag ingredients with diet restrictions and nutrition goals (local USDA + FoodOn).

Usage:
  uv run python scripts/build_foodon_index_cache.py
  uv run python scripts/tag_ingredients.py
  uv run python scripts/tag_ingredients.py --no-foodon --limit 5000
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

from diet_tags_core import flatten_ingredient_rows, load_diet_tags, tag_ingredient
from diet_tags_io import load_foods_catalog, load_nutrients_for_fdc, write_table

OUT_DIR = ROOT / "scratch" / "tag"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag ingredients from diet_tags.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-foodon", action="store_true")
    parser.add_argument("--foodon-min-score", type=float, default=0.55)
    parser.add_argument("--tags-path", type=Path, default=None)
    args = parser.parse_args()

    registry = load_diet_tags(args.tags_path)
    foods = load_foods_catalog(limit=args.limit)

    foodon_index = None
    if not args.no_foodon:
        from foodon_index import FoodOnIndex

        foodon_index = FoodOnIndex.from_owl()

    fdc_ids = set(foods["fdc_id"].astype(int).tolist())
    nutrient_ids = {spec.nutrient_id for spec in registry.nutrients.values()}
    for spec in registry.nutrients.values():
        if spec.fallback_nutrient_id is not None:
            nutrient_ids.add(spec.fallback_nutrient_id)

    print(f"Loading nutrients for {len(fdc_ids):,} foods...", flush=True)
    nutrient_lookup = load_nutrients_for_fdc(fdc_ids, nutrient_ids)
    print(f"Nutrient values loaded: {len(nutrient_lookup):,}", flush=True)

    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    foodon_maps: list[dict] = []

    for row in foods.itertuples(index=False):
        fdc_id = int(row.fdc_id)
        desc = str(row.description)
        ing = str(row.ingredients) if row.ingredients is not None and pd.notna(row.ingredients) else None
        result = tag_ingredient(
            fdc_id,
            desc,
            ing,
            registry,
            nutrient_lookup=nutrient_lookup,
            foodon_index=foodon_index,
            foodon_min_score=args.foodon_min_score,
        )
        long_rows.extend(flatten_ingredient_rows(result))
        wide = {"fdc_id": fdc_id, "description": desc}
        for tslug, val in result["tags"].items():
            if val is not None:
                wide[f"tag_{tslug}"] = bool(val)
        for cslug in result["contains"]:
            wide[f"contains_{cslug}"] = True
        wide_rows.append(wide)

        if foodon_index is not None:
            match = foodon_index.best_match(desc, min_score=args.foodon_min_score)
            if match:
                foodon_maps.append(
                    {
                        "fdc_id": fdc_id,
                        "description": desc,
                        "foodon_id": match["id"],
                        "foodon_label": match["label"],
                        "score": match["score"],
                    }
                )

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    write_table(long_df, OUT_DIR / "ingredient_diet_tags_long.parquet")
    write_table(wide_df, OUT_DIR / "ingredient_diet_tags_wide.parquet")
    if foodon_maps:
        write_table(pd.DataFrame(foodon_maps), OUT_DIR / "ingredient_foodon_map.parquet")

    summary = {
        "foods_scanned": len(foods),
        "long_rows": len(long_df),
        "tag_true_counts": {
            col.replace("tag_", ""): int(wide_df[col].sum())
            for col in wide_df.columns
            if col.startswith("tag_")
        },
    }
    (OUT_DIR / "ingredient_diet_tags_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Foods scanned: {len(foods):,}", flush=True)
    print(f"Wrote {OUT_DIR / 'ingredient_diet_tags_long.parquet'}", flush=True)
    print(f"Wrote {OUT_DIR / 'ingredient_diet_tags_wide.parquet'}", flush=True)
    if summary["tag_true_counts"]:
        top = sorted(summary["tag_true_counts"].items(), key=lambda x: -x[1])[:8]
        print("Top tag counts:", flush=True)
        for slug, n in top:
            print(f"  {slug}: {n:,}", flush=True)


if __name__ == "__main__":
    main()
