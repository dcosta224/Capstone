#!/usr/bin/env python3
"""Aggregate per-100g USDA nutrients into wide recipe.recipe_nutrients rows.

Prerequisite: recipe.resolved_recipes populated (load_resolved_recipes.py --execute).

Usage:
  uv run python scripts/load_recipe_nutrients.py --dry-run
  uv run python scripts/load_recipe_nutrients.py --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from recipe_nutrient_columns import assign_nutrient_columns, build_recipe_nutrients_ddl


def fetch_nutrient_catalog(cur) -> list[dict]:
    cur.execute(
        "SELECT id, name, unit_name FROM usda.nutrient ORDER BY id"
    )
    return [
        {"id": int(r[0]), "name": r[1], "unit_name": r[2]}
        for r in cur.fetchall()
    ]


def fetch_resolved_ingredients(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT recipe_id, recipe_name, ingredient_idx, fdc_id, gram_weight
        FROM recipe.resolved_recipes
        WHERE fdc_id IS NOT NULL AND gram_weight IS NOT NULL
        ORDER BY recipe_id, ingredient_idx
        """
    )
    cols = ["recipe_id", "recipe_name", "ingredient_idx", "fdc_id", "gram_weight"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_food_nutrients(cur, fdc_ids: list[int]) -> pd.DataFrame:
    if not fdc_ids:
        return pd.DataFrame(columns=["fdc_id", "nutrient_id", "amount"])
    cur.execute(
        """
        SELECT fdc_id, nutrient_id, amount
        FROM usda.food_nutrient
        WHERE fdc_id = ANY(%s) AND amount IS NOT NULL
        """,
        (fdc_ids,),
    )
    return pd.DataFrame(cur.fetchall(), columns=["fdc_id", "nutrient_id", "amount"])


def build_recipe_nutrient_wide(
    ingredients: pd.DataFrame,
    food_nutrients: pd.DataFrame,
    nutrient_col_map: dict[int, str],
) -> pd.DataFrame:
    ing = ingredients.copy()
    ing["fdc_id"] = ing["fdc_id"].astype(int)
    ing["gram_weight"] = ing["gram_weight"].astype(float)

    fn = food_nutrients.copy()
    fn["fdc_id"] = fn["fdc_id"].astype(int)
    fn["nutrient_id"] = fn["nutrient_id"].astype(int)
    fn["amount"] = fn["amount"].astype(float)

    merged = ing.merge(fn, on="fdc_id", how="inner")
    merged["contrib"] = merged["amount"] * (merged["gram_weight"] / 100.0)
    merged["col_name"] = merged["nutrient_id"].map(nutrient_col_map)

    agg = (
        merged.groupby(["recipe_id", "recipe_name", "col_name"], as_index=False)["contrib"]
        .sum()
    )
    wide = agg.pivot_table(
        index=["recipe_id", "recipe_name"],
        columns="col_name",
        values="contrib",
        aggfunc="sum",
    ).reset_index()

    meta = (
        ingredients.groupby(["recipe_id", "recipe_name"], as_index=False)
        .agg(n_ingredients=("ingredient_idx", "count"), total_gram_weight=("gram_weight", "sum"))
    )
    return meta.merge(wide, on=["recipe_id", "recipe_name"], how="left")


def recreate_table(cur, ddl: str) -> None:
    cur.execute("DROP TABLE IF EXISTS recipe.recipe_nutrients CASCADE")
    cur.execute(ddl)


def insert_wide(cur, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    sql = f"INSERT INTO recipe.recipe_nutrients ({col_list}) VALUES %s"
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    psycopg2.extras.execute_values(cur, sql, records, page_size=200)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load recipe.recipe_nutrients")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        parser.error("Specify --dry-run or --execute")

    load_dotenv()
    conn = connect()
    try:
        with conn.cursor() as cur:
            catalog = fetch_nutrient_catalog(cur)
            nutrient_col_map = assign_nutrient_columns(catalog)
            ingredients = fetch_resolved_ingredients(cur)

        if ingredients.empty:
            print("recipe.resolved_recipes is empty — run load_resolved_recipes.py --execute first", flush=True)
            sys.exit(1)

        fdc_ids = sorted(ingredients["fdc_id"].astype(int).unique().tolist())
        with conn.cursor() as cur:
            food_nutrients = fetch_food_nutrients(cur, fdc_ids)

        wide = build_recipe_nutrient_wide(ingredients, food_nutrients, nutrient_col_map)
        n_nutrient_cols = len(nutrient_col_map)
        print(f"Recipes: {len(wide)}", flush=True)
        print(f"Nutrient columns: {n_nutrient_cols}", flush=True)
        print(f"Ingredient lines used: {len(ingredients)}", flush=True)
        print(f"food_nutrient rows matched: {len(food_nutrients):,}", flush=True)

        # sample non-null macro columns if present
        for col in ("energy_kcal", "protein_g", "carbohydrate_by_difference_g"):
            if col in wide.columns:
                nn = wide[col].notna().sum()
                print(f"  {col} non-null: {nn}/{len(wide)}", flush=True)

        if args.dry_run:
            print("\nSample (first recipe, first 5 nutrient cols):", flush=True)
            nut_cols = [c for c in wide.columns if c not in ("recipe_id", "recipe_name", "n_ingredients", "total_gram_weight")][:5]
            print(wide[["recipe_id", "recipe_name", "total_gram_weight"] + nut_cols].head(1).to_string(), flush=True)
            return

        ddl = build_recipe_nutrients_ddl(nutrient_col_map)
        with conn.cursor() as cur:
            recreate_table(cur, ddl)
            n = insert_wide(cur, wide)
        conn.commit()
        print(f"Loaded {n} rows into recipe.recipe_nutrients", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
