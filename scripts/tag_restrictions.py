#!/usr/bin/env python3
"""Load ingredient and recipe restriction tags from allergen taxonomy.

Usage:
  uv run python scripts/tag_restrictions.py --dry-run
  uv run python scripts/tag_restrictions.py --execute
  uv run python scripts/tag_restrictions.py --execute --foodon
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from db import connect, load_dotenv
from tag_restrictions_core import ingredient_restriction_rows, load_taxonomy


def fetch_food_4macro_descriptions(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT f.fdc_id, f.description, bf.ingredients AS branded_ingredients
        FROM usda.food_4macro f
        LEFT JOIN usda.branded_food bf ON bf.fdc_id = f.fdc_id
        ORDER BY f.fdc_id
        """
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=["fdc_id", "description", "branded_ingredients"],
    )


def fetch_resolved_lines(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT recipe_id, ingredient_idx, fdc_id
        FROM recipe.resolved_recipes
        WHERE fdc_id IS NOT NULL
        ORDER BY recipe_id, ingredient_idx
        """
    )
    return pd.DataFrame(cur.fetchall(), columns=["recipe_id", "ingredient_idx", "fdc_id"])


def insert_rows(cur, table: str, rows: list[dict], columns: list[str]) -> int:
    if not rows:
        return 0
    col_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s"
    records = [tuple(r[c] for c in columns) for r in rows]
    psycopg2.extras.execute_values(cur, sql, records, page_size=500)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load tag.ingredient_restriction and tag.recipe_restriction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument("--foodon", action="store_true", help="Enable FoodOn ancestry matching")
    parser.add_argument("--foodon-min-score", type=float, default=0.55)
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        parser.error("Specify --dry-run or --execute")

    rules, universal = load_taxonomy(args.taxonomy)

    foodon_index = None
    if args.foodon:
        from foodon_index import FoodOnIndex

        foodon_index = FoodOnIndex.from_owl()

    load_dotenv()
    conn = connect()
    try:
        with conn.cursor() as cur:
            foods = fetch_food_4macro_descriptions(cur)
            resolved = fetch_resolved_lines(cur)

        ing_rows: list[dict] = []
        for row in foods.itertuples(index=False):
            ing_rows.extend(
                ingredient_restriction_rows(
                    int(row.fdc_id),
                    str(row.description or ""),
                    str(row.branded_ingredients) if row.branded_ingredients else None,
                    rules,
                    universal=universal,
                    foodon_index=foodon_index,
                    foodon_min_score=args.foodon_min_score,
                )
            )

        fdc_to_slugs: dict[int, set[str]] = defaultdict(set)
        for r in ing_rows:
            fdc_to_slugs[int(r["fdc_id"])].add(str(r["restriction_slug"]))

        recipe_rows: list[dict] = []
        if not resolved.empty:
            for recipe_id, grp in resolved.groupby("recipe_id"):
                slugs: set[str] = set()
                for fdc_id in grp["fdc_id"].astype(int):
                    slugs |= fdc_to_slugs.get(int(fdc_id), set())
                for slug in slugs:
                    n_lines = sum(
                        1
                        for fdc_id in grp["fdc_id"].astype(int)
                        if slug in fdc_to_slugs.get(int(fdc_id), set())
                    )
                    recipe_rows.append(
                        {
                            "recipe_id": int(recipe_id),
                            "restriction_slug": slug,
                            "n_triggering_lines": n_lines,
                        }
                    )

        slug_counts = defaultdict(int)
        for r in ing_rows:
            slug_counts[r["restriction_slug"]] += 1

        print(f"Foods scanned: {len(foods):,}", flush=True)
        print(f"Ingredient restriction rows: {len(ing_rows):,}", flush=True)
        print(f"Recipe restriction rows: {len(recipe_rows):,}", flush=True)
        print("Hits by restriction:", flush=True)
        for slug, n in sorted(slug_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {slug}: {n:,}", flush=True)

        if args.dry_run:
            if ing_rows:
                print("\nSample:", ing_rows[0], flush=True)
            return

        with conn.cursor() as cur:
            cur.execute("TRUNCATE tag.ingredient_restriction, tag.recipe_restriction")
            insert_rows(
                cur,
                "tag.ingredient_restriction",
                ing_rows,
                ["fdc_id", "restriction_slug", "source", "matched_term"],
            )
            insert_rows(
                cur,
                "tag.recipe_restriction",
                recipe_rows,
                ["recipe_id", "restriction_slug", "n_triggering_lines"],
            )
        conn.commit()
        print("Restriction tags loaded.", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
