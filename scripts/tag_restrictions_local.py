#!/usr/bin/env python3
"""Tag ingredient restrictions locally from USDA CSVs (no Supabase).

Reads:
  - Data/All_Food_Data_April_2026/food.csv
  - Data/All_Food_Data_April_2026/branded_food.csv (optional)
  - scratch/food_4macro.csv (optional filter; if missing, uses all foods)

Writes:
  - scratch/tag/ingredient_restrictions.parquet
  - scratch/tag/ingredient_foodon_map.parquet (when --foodon)

Usage:
  uv run python scripts/build_foodon_index_cache.py   # first time (~1-2 min)
  uv run python scripts/tag_restrictions_local.py
  uv run python scripts/tag_restrictions_local.py --foodon --limit 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_paths import USDA_BRANDED_CSV, USDA_FOOD_CSV
from tag_restrictions_core import ingredient_restriction_rows, load_taxonomy

OUT_DIR = ROOT / "scratch" / "tag"
FOOD_4MACRO_CACHE = ROOT / "scratch" / "food_4macro.csv"


def load_foods(*, limit: int | None = None) -> pd.DataFrame:
    if not USDA_FOOD_CSV.is_file():
        raise FileNotFoundError(f"Missing {USDA_FOOD_CSV}")

    if FOOD_4MACRO_CACHE.is_file():
        catalog = pd.read_csv(FOOD_4MACRO_CACHE, usecols=["fdc_id"])
        fdc_ids = set(catalog["fdc_id"].astype(int).tolist())
        foods = pd.read_csv(
            USDA_FOOD_CSV,
            usecols=["fdc_id", "description"],
            dtype={"description": "string"},
        )
        foods = foods[foods["fdc_id"].isin(fdc_ids)]
    else:
        foods = pd.read_csv(
            USDA_FOOD_CSV,
            usecols=["fdc_id", "description"],
            dtype={"description": "string"},
            nrows=limit,
        )

    if limit is not None and FOOD_4MACRO_CACHE.is_file():
        foods = foods.head(limit)

    branded = None
    if USDA_BRANDED_CSV.is_file():
        branded = pd.read_csv(
            USDA_BRANDED_CSV,
            usecols=["fdc_id", "ingredients"],
            dtype={"ingredients": "string"},
        )

    if branded is not None:
        foods = foods.merge(branded, on="fdc_id", how="left")
    else:
        foods["ingredients"] = None

    foods["description"] = foods["description"].fillna("").astype(str)
    return foods


def main() -> None:
    parser = argparse.ArgumentParser(description="Local restriction tagging from USDA CSVs")
    parser.add_argument("--foodon", action="store_true", help="Enable FoodOn ancestry matching")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (when no food_4macro cache)")
    parser.add_argument("--foodon-min-score", type=float, default=0.55)
    args = parser.parse_args()

    rules, universal = load_taxonomy()
    foods = load_foods(limit=args.limit)

    foodon_index = None
    if args.foodon:
        from foodon_index import FoodOnIndex

        foodon_index = FoodOnIndex.from_owl()

    rows: list[dict] = []
    foodon_maps: list[dict] = []
    for row in foods.itertuples(index=False):
        fdc_id = int(row.fdc_id)
        desc = str(row.description)
        ing_text = str(row.ingredients) if row.ingredients is not None and pd.notna(row.ingredients) else None
        rows.extend(
            ingredient_restriction_rows(
                fdc_id,
                desc,
                ing_text,
                rules,
                universal=universal,
                foodon_index=foodon_index,
                foodon_min_score=args.foodon_min_score,
            )
        )
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    restr_df = pd.DataFrame(rows)
    restr_path = OUT_DIR / "ingredient_restrictions.parquet"
    restr_df.to_parquet(restr_path, index=False)

    print(f"Foods scanned: {len(foods):,}", flush=True)
    print(f"Restriction rows: {len(restr_df):,}", flush=True)
    print(f"Wrote {restr_path}", flush=True)

    if foodon_maps:
        map_df = pd.DataFrame(foodon_maps)
        map_path = OUT_DIR / "ingredient_foodon_map.parquet"
        map_df.to_parquet(map_path, index=False)
        print(f"FoodOn mappings: {len(map_df):,}", flush=True)
        print(f"Wrote {map_path}", flush=True)

    if not restr_df.empty:
        print("\nTop restrictions:", flush=True)
        print(restr_df["restriction_slug"].value_counts().head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
