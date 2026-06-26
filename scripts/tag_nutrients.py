#!/usr/bin/env python3
"""Load ingredient and recipe nutrient tags into the tag schema.

Prerequisites:
  - sql/14_create_tag_schema.sql applied
  - recipe.resolved_recipes populated

Usage:
  uv run python scripts/tag_nutrients.py --dry-run
  uv run python scripts/tag_nutrients.py --execute
  uv run python scripts/tag_nutrients.py --execute --catalog-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from tag_dimensions import NUTRIENT_DIMENSIONS, TAG_NUTRIENT_IDS
from tag_nutrient_core import (
    add_corpus_labels,
    build_ingredient_nutrient_rows,
    build_recipe_nutrient_wide,
    nutrient_lookup_from_df,
)


def seed_dimensions(cur) -> dict[str, int]:
    ids: dict[str, int] = {}
    for dim in NUTRIENT_DIMENSIONS:
        cur.execute(
            """
            INSERT INTO tag.dimension (
                slug, nutrient_id, unit, direction, stories,
                dv_per_serving, low_dv_frac, high_dv_frac, fallback_nutrient_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (slug) DO UPDATE SET
                nutrient_id = EXCLUDED.nutrient_id,
                unit = EXCLUDED.unit,
                direction = EXCLUDED.direction,
                stories = EXCLUDED.stories,
                dv_per_serving = EXCLUDED.dv_per_serving,
                low_dv_frac = EXCLUDED.low_dv_frac,
                high_dv_frac = EXCLUDED.high_dv_frac,
                fallback_nutrient_id = EXCLUDED.fallback_nutrient_id
            RETURNING id
            """,
            (
                dim.slug,
                dim.nutrient_id,
                dim.unit,
                dim.direction,
                list(dim.stories),
                dim.dv_per_serving,
                dim.low_dv_frac,
                dim.high_dv_frac,
                dim.fallback_nutrient_id,
            ),
        )
        ids[dim.slug] = int(cur.fetchone()[0])
    return ids


def fetch_food_4macro_ids(cur) -> list[int]:
    cur.execute("SELECT fdc_id FROM usda.food_4macro ORDER BY fdc_id")
    return [int(r[0]) for r in cur.fetchall()]


def fetch_resolved_ingredients(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT recipe_id, recipe_name, ingredient_idx, fdc_id, gram_weight
        FROM recipe.resolved_recipes
        WHERE fdc_id IS NOT NULL AND gram_weight IS NOT NULL
        ORDER BY recipe_id, ingredient_idx
        """
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=["recipe_id", "recipe_name", "ingredient_idx", "fdc_id", "gram_weight"],
    )


def fetch_food_nutrients(cur, fdc_ids: list[int]) -> pd.DataFrame:
    if not fdc_ids:
        return pd.DataFrame(columns=["fdc_id", "nutrient_id", "amount"])
    cur.execute(
        """
        SELECT fdc_id, nutrient_id, amount
        FROM usda.food_nutrient
        WHERE fdc_id = ANY(%s)
          AND nutrient_id = ANY(%s)
          AND amount IS NOT NULL
        """,
        (fdc_ids, list(TAG_NUTRIENT_IDS)),
    )
    return pd.DataFrame(cur.fetchall(), columns=["fdc_id", "nutrient_id", "amount"])


def insert_rows(cur, table: str, rows: list[dict], columns: list[str]) -> int:
    if not rows:
        return 0
    col_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s"
    records = [tuple(r[c] for c in columns) for r in rows]
    psycopg2.extras.execute_values(cur, sql, records, page_size=500)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load tag.ingredient_nutrient and tag.recipe_nutrient")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Only tag food_4macro ingredients (skip recipe rollup)",
    )
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        parser.error("Specify --dry-run or --execute")

    load_dotenv()
    conn = connect()
    try:
        with conn.cursor() as cur:
            dimension_ids = seed_dimensions(cur)
            catalog_ids = fetch_food_4macro_ids(cur)
            ingredients = pd.DataFrame() if args.catalog_only else fetch_resolved_ingredients(cur)

        recipe_fdc = (
            sorted(ingredients["fdc_id"].astype(int).unique().tolist()) if not ingredients.empty else []
        )
        all_fdc = sorted(set(catalog_ids) | set(recipe_fdc))

        with conn.cursor() as cur:
            food_nutrients = fetch_food_nutrients(cur, all_fdc)

        lookup = nutrient_lookup_from_df(food_nutrients)
        ing_rows = build_ingredient_nutrient_rows(catalog_ids, lookup, dimension_ids)

        recipe_df = pd.DataFrame()
        percentile_refs: list[dict] = []
        if not ingredients.empty:
            recipe_df = build_recipe_nutrient_wide(ingredients, lookup, dimension_ids)
            recipe_df, percentile_refs = add_corpus_labels(recipe_df)

        print(f"food_4macro foods: {len(catalog_ids):,}", flush=True)
        print(f"Ingredient nutrient rows: {len(ing_rows):,}", flush=True)
        print(f"Resolved ingredient lines: {len(ingredients):,}", flush=True)
        print(f"Recipe dimension rows: {len(recipe_df):,}", flush=True)

        if not recipe_df.empty:
            for slug in {d.slug for d in NUTRIENT_DIMENSIONS}:
                sub = recipe_df[recipe_df["dimension_slug"] == slug]
                if sub.empty:
                    print(f"  {slug}: no recipe coverage", flush=True)
                else:
                    nn = sub["absolute_per_serving"].notna().sum()
                    print(f"  {slug}: {nn} recipes tagged", flush=True)

        if args.dry_run:
            if ing_rows:
                print("\nSample ingredient row:", ing_rows[0], flush=True)
            if not recipe_df.empty:
                print("\nSample recipe row:", recipe_df.iloc[0].to_dict(), flush=True)
            return

        with conn.cursor() as cur:
            cur.execute("TRUNCATE tag.ingredient_nutrient")
            insert_rows(
                cur,
                "tag.ingredient_nutrient",
                ing_rows,
                ["fdc_id", "dimension_id", "absolute_per_100g", "nutrient_id_used"],
            )
            if not recipe_df.empty:
                cur.execute("TRUNCATE tag.recipe_nutrient, tag.corpus_percentile")
                recipe_insert = [
                    {
                        "recipe_id": int(r.recipe_id),
                        "dimension_id": int(r.dimension_id),
                        "absolute_total": float(r.absolute_total),
                        "absolute_per_serving": float(r.absolute_per_serving),
                        "corpus_percentile": float(r.corpus_percentile)
                        if pd.notna(r.corpus_percentile)
                        else None,
                        "absolute_label": r.absolute_label,
                        "corpus_label": r.corpus_label,
                        "nutrient_id_used": int(r.nutrient_id_used)
                        if r.nutrient_id_used is not None
                        else None,
                        "n_ingredients_with_value": int(r.n_ingredients_with_value),
                    }
                    for r in recipe_df.itertuples(index=False)
                ]
                insert_rows(
                    cur,
                    "tag.recipe_nutrient",
                    recipe_insert,
                    [
                        "recipe_id",
                        "dimension_id",
                        "absolute_total",
                        "absolute_per_serving",
                        "corpus_percentile",
                        "absolute_label",
                        "corpus_label",
                        "nutrient_id_used",
                        "n_ingredients_with_value",
                    ],
                )
                insert_rows(
                    cur,
                    "tag.corpus_percentile",
                    percentile_refs,
                    ["dimension_id", "percentile", "value", "n_recipes"],
                )
        conn.commit()
        print("Tag tables loaded.", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
