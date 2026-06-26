#!/usr/bin/env python3
"""Run dietary tagging EDA queries and print a summary report."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from tag_dimensions import TAG_NUTRIENT_IDS

COVERAGE_SQL = """
SELECT n.id, n.name, n.unit_name,
       COUNT(DISTINCT fn.fdc_id) AS n_foods,
       ROUND(100.0 * COUNT(DISTINCT fn.fdc_id) /
         NULLIF((SELECT COUNT(*) FROM usda.food_4macro), 0), 2) AS pct_of_catalog
FROM usda.nutrient n
LEFT JOIN usda.food_nutrient fn
  ON fn.nutrient_id = n.id
 AND fn.fdc_id IN (SELECT fdc_id FROM usda.food_4macro)
 AND fn.amount IS NOT NULL
WHERE n.id = ANY(%s)
GROUP BY n.id, n.name, n.unit_name
ORDER BY n.id
"""

RESOLVED_SQL = """
SELECT fn.nutrient_id, COUNT(*) AS n_lines_with_value,
       COUNT(DISTINCT rr.recipe_id) AS n_recipes
FROM recipe.resolved_recipes rr
JOIN usda.food_nutrient fn ON fn.fdc_id = rr.fdc_id AND fn.amount IS NOT NULL
WHERE rr.fdc_id IS NOT NULL
  AND fn.nutrient_id = ANY(%s)
GROUP BY fn.nutrient_id
ORDER BY fn.nutrient_id
"""

BRANDED_SQL = """
SELECT
  COUNT(*) AS n_food_4macro,
  COUNT(bf.fdc_id) AS n_with_branded_row,
  COUNT(bf.ingredients) FILTER (WHERE bf.ingredients IS NOT NULL AND bf.ingredients <> '') AS n_with_ingredients_text
FROM usda.food_4macro f
LEFT JOIN usda.branded_food bf ON bf.fdc_id = f.fdc_id
"""


def main() -> None:
    load_dotenv()
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM usda.food_4macro")
            n_catalog = int(cur.fetchone()[0])

            cur.execute(COVERAGE_SQL, (list(TAG_NUTRIENT_IDS),))
            cov_cols = ["id", "name", "unit_name", "n_foods", "pct_of_catalog"]
            cov = pd.DataFrame(cur.fetchall(), columns=cov_cols)

            try:
                cur.execute("SELECT COUNT(*) FROM recipe.resolved_recipes")
                n_resolved = int(cur.fetchone()[0])
                cur.execute(RESOLVED_SQL, (list(TAG_NUTRIENT_IDS),))
                res = pd.DataFrame(
                    cur.fetchall(),
                    columns=["nutrient_id", "n_lines_with_value", "n_recipes"],
                )
            except Exception:
                conn.rollback()
                n_resolved = 0
                res = pd.DataFrame()

            cur.execute(BRANDED_SQL)
            branded = pd.DataFrame(
                [cur.fetchone()],
                columns=["n_food_4macro", "n_with_branded_row", "n_with_ingredients_text"],
            )

        print("=== Dietary Tagging EDA ===\n", flush=True)
        print(f"food_4macro rows: {n_catalog:,}", flush=True)
        print(f"resolved_recipes lines: {n_resolved:,}\n", flush=True)

        print("Nutrient coverage on food_4macro:")
        print(cov.to_string(index=False), flush=True)

        if not res.empty:
            print("\nNutrient coverage on resolved recipe lines:")
            print(res.to_string(index=False), flush=True)

        print("\nBranded ingredients text coverage:")
        print(branded.to_string(index=False), flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
