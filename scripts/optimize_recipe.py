#!/usr/bin/env python3
"""CLI: optimize a single resolved recipe's portions toward macro bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from mvp_data import build_recipe_macro_inputs, fetch_food_nutrients_for_recipe, fetch_resolved_ingredients
from recipe_macro_optimizer import (
    IngredientMeta,
    MacroBounds,
    OptimizerConfig,
    RecipeMacroOptimizer,
    derive_macro_bounds_from_fractions,
    format_serving_display,
    macros_to_dict,
)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Optimize recipe portions")
    parser.add_argument("--recipe-id", type=int, required=True)
    parser.add_argument("--kcal-min", type=float, default=400)
    parser.add_argument("--kcal-max", type=float, default=600)
    parser.add_argument("--fat-frac-min", type=float, default=0.25)
    parser.add_argument("--fat-frac-max", type=float, default=0.35)
    parser.add_argument("--carb-frac-min", type=float, default=0.45)
    parser.add_argument("--carb-frac-max", type=float, default=0.55)
    parser.add_argument("--protein-frac-min", type=float, default=0.15)
    parser.add_argument("--protein-frac-max", type=float, default=0.25)
    args = parser.parse_args()

    conn = connect()
    try:
        with conn.cursor() as cur:
            ingredients = fetch_resolved_ingredients(cur, args.recipe_id)
            fdc_ids = [int(x) for x in ingredients["fdc_id"].dropna().astype(int).tolist()]
            food_nutrients = fetch_food_nutrients_for_recipe(cur, fdc_ids)
    finally:
        conn.close()

    x0, M = build_recipe_macro_inputs(ingredients, food_nutrients)
    bounds = derive_macro_bounds_from_fractions(
        args.kcal_min,
        args.kcal_max,
        args.fat_frac_min,
        args.fat_frac_max,
        args.carb_frac_min,
        args.carb_frac_max,
        args.protein_frac_min,
        args.protein_frac_max,
    )
    result = RecipeMacroOptimizer().optimize(x0, M, OptimizerConfig(macro_bounds=bounds))
    meta = [
        IngredientMeta(
            ingredient_idx=int(row.ingredient_idx),
            ingredient=str(row.ingredient),
            fdc_id=int(row.fdc_id) if row.fdc_id is not None else None,
            quantity=float(row.quantity) if row.quantity is not None else None,
            unit=str(row.unit) if row.unit else None,
            portion_label=str(row.portion_label) if row.portion_label else None,
            fdc_description=str(row.fdc_description) if row.fdc_description else None,
        )
        for row in ingredients.itertuples(index=False)
    ]
    out = {
        "recipe_id": args.recipe_id,
        "macros_before": macros_to_dict(result.macros_before),
        "macros_after": macros_to_dict(result.macros_after),
        "portion_score": result.portion_score,
        "avg_pct_change": result.avg_pct_change,
        "max_pct_change": result.max_pct_change,
        "ingredients": format_serving_display(result, x0, meta),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
