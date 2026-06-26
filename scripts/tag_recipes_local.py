#!/usr/bin/env python3
"""Roll up ingredient diet tags to recipe level (local files).

Requires:
  - scratch/tag/ingredient_diet_tags_wide.parquet (from tag_ingredients.py)
  - Resolved recipe lines (parquet/csv) with recipe_id, recipe_name, fdc_id, gram_weight

Usage:
  uv run python scripts/tag_recipes_local.py --resolved scratch/path/to/lines.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diet_tags_core import load_diet_tags, tag_recipe
from diet_tags_io import load_nutrients_for_fdc, write_table

OUT_DIR = ROOT / "scratch" / "tag"
INGREDIENT_WIDE = OUT_DIR / "ingredient_diet_tags_wide.parquet"
INGREDIENT_WIDE_CSV = OUT_DIR / "ingredient_diet_tags_wide.csv"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_ingredient_wide() -> pd.DataFrame:
    if INGREDIENT_WIDE.is_file():
        return pd.read_parquet(INGREDIENT_WIDE)
    if INGREDIENT_WIDE_CSV.is_file():
        return pd.read_csv(INGREDIENT_WIDE_CSV)
    raise FileNotFoundError("Run tag_ingredients.py first.")


def _ingredient_row_from_wide(wide_row: pd.Series, registry) -> dict:
    contains_set = {
        col.replace("contains_", "")
        for col in wide_row.index
        if col.startswith("contains_") and bool(wide_row[col])
    }
    tags = {}
    for tag_slug in registry.ingredient_tags:
        col = f"tag_{tag_slug}"
        if col in wide_row.index and pd.notna(wide_row[col]):
            tags[tag_slug] = bool(wide_row[col])
    return {"contains_set": contains_set, "tags": tags}


def _recipe_nutrient_totals(
    lines: pd.DataFrame,
    nutrient_lookup: dict[tuple[int, int], float],
    registry,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in lines.itertuples(index=False):
        grams = float(row.gram_weight)
        fdc_id = int(row.fdc_id)
        for slug, spec in registry.nutrients.items():
            val = nutrient_lookup.get((fdc_id, spec.nutrient_id))
            if val is None and spec.fallback_nutrient_id is not None:
                val = nutrient_lookup.get((fdc_id, spec.fallback_nutrient_id))
            if val is None:
                continue
            contrib = val * (grams / 100.0)
            totals[slug] = totals.get(slug, 0.0) + contrib
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Recipe-level diet tag rollup")
    parser.add_argument(
        "--resolved",
        type=Path,
        required=True,
        help="Parquet/CSV with recipe_id, recipe_name, fdc_id, gram_weight",
    )
    parser.add_argument("--tags-path", type=Path, default=None)
    args = parser.parse_args()

    registry = load_diet_tags(args.tags_path)
    resolved = _read_table(args.resolved)
    required = {"recipe_id", "fdc_id", "gram_weight"}
    missing = required - set(resolved.columns)
    if missing:
        raise ValueError(f"Resolved file missing columns: {sorted(missing)}")

    if "recipe_name" not in resolved.columns:
        resolved["recipe_name"] = resolved["recipe_id"].astype(str)

    wide = _load_ingredient_wide()
    wide_by_fdc = {int(r.fdc_id): r for r in wide.itertuples(index=False)}

    fdc_ids = set(resolved["fdc_id"].astype(int).tolist())
    nutrient_ids = {spec.nutrient_id for spec in registry.nutrients.values()}
    for spec in registry.nutrients.values():
        if spec.fallback_nutrient_id is not None:
            nutrient_ids.add(spec.fallback_nutrient_id)
    nutrient_lookup = load_nutrients_for_fdc(fdc_ids, nutrient_ids)

    recipe_rows: list[dict] = []
    wide_out: list[dict] = []

    for recipe_id, grp in resolved.groupby("recipe_id"):
        recipe_name = str(grp["recipe_name"].iloc[0])
        ing_rows: list[dict] = []
        for line in grp.itertuples(index=False):
            fdc_id = int(line.fdc_id)
            w = wide_by_fdc.get(fdc_id)
            if w is not None:
                ing_rows.append(_ingredient_row_from_wide(w, registry))
            else:
                ing_rows.append({"contains_set": set(), "tags": {}})

        totals = _recipe_nutrient_totals(grp, nutrient_lookup, registry)
        result = tag_recipe(
            int(recipe_id),
            recipe_name,
            ing_rows,
            registry,
            nutrient_totals_per_serving=totals,
        )
        recipe_rows.append(result)
        row = {"recipe_id": int(recipe_id), "recipe_name": recipe_name}
        for c in result["contains_union"]:
            row[f"contains_{c}"] = True
        for tslug, val in result["tags"].items():
            if val is not None:
                row[f"tag_{tslug}"] = bool(val)
        for nslug, val in totals.items():
            row[f"nutrient_{nslug}"] = val
        wide_out.append(row)

    wide_df = pd.DataFrame(wide_out)
    write_table(wide_df, OUT_DIR / "recipe_diet_tags_wide.parquet")

    summary = {
        "recipes": len(wide_df),
        "tag_true_counts": {
            col.replace("tag_", ""): int(wide_df[col].sum())
            for col in wide_df.columns
            if col.startswith("tag_") and wide_df[col].dtype == bool
        },
    }
    (OUT_DIR / "recipe_diet_tags_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Recipes tagged: {len(wide_df):,}", flush=True)
    print(f"Wrote {OUT_DIR / 'recipe_diet_tags_wide.parquet'}", flush=True)


if __name__ == "__main__":
    main()
