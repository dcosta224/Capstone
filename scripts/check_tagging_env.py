#!/usr/bin/env python3
"""Verify prerequisites for the dietary tagging pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_paths import FOODON_INDEX_CACHE, FOODON_OWL, USDA_FOOD_CSV


def check_local_files() -> list[str]:
    issues: list[str] = []
    ok: list[str] = []

    if FOODON_OWL.is_file():
        ok.append(f"FoodOn OWL: {FOODON_OWL.relative_to(ROOT)}")
    else:
        issues.append(f"Missing FoodOn OWL at {FOODON_OWL.relative_to(ROOT)}")

    if (ROOT / "foodon_web" / "foodon_index.py").is_file():
        ok.append("foodon_web/foodon_index.py")
    else:
        issues.append("Missing foodon_web/foodon_index.py")

    if FOODON_INDEX_CACHE.is_file():
        ok.append(f"Index cache: {FOODON_INDEX_CACHE.relative_to(ROOT)}")
    else:
        issues.append(
            "No index cache yet — run: uv run python scripts/build_foodon_index_cache.py"
        )

    if USDA_FOOD_CSV.is_file():
        ok.append(f"USDA food.csv: {USDA_FOOD_CSV.relative_to(ROOT)}")
    else:
        issues.append(f"USDA CSV not found (optional for DB path): {USDA_FOOD_CSV}")

    for line in ok:
        print(f"  OK: {line}", flush=True)
    return issues


def check_database() -> tuple[list[str], dict[str, int | None]]:
    from db import connect, load_dotenv

    load_dotenv()
    counts: dict[str, int | None] = {}
    issues: list[str] = []

    env_path = ROOT / ".env"
    if not env_path.is_file():
        issues.append("No .env file — skip DB checks or copy from .env.example for Supabase")
        return issues, counts

    try:
        conn = connect()
    except Exception as exc:
        issues.append(f"Database connection failed: {exc}")
        return issues, counts

    tables = (
        ("usda.food_4macro", "food_4macro catalog"),
        ("usda.food_nutrient", "food_nutrient"),
        ("usda.nutrient", "nutrient reference"),
        ("recipe.resolved_recipes", "resolved recipe lines"),
        ("recipe.recipe_nutrients", "recipe nutrient rollups"),
        ("tag.dimension", "tag dimension registry"),
    )
    try:
        with conn.cursor() as cur:
            for table, label in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = int(cur.fetchone()[0])
                except Exception:
                    conn.rollback()
                    counts[table] = None
                    if table.startswith("tag."):
                        issues.append(f"Table {table} not created — run sql/14_create_tag_schema.sql")
                    else:
                        issues.append(f"Table {table} missing or inaccessible")
    finally:
        conn.close()

    if counts.get("recipe.resolved_recipes") == 0:
        issues.append("recipe.resolved_recipes is empty — run load_resolved_recipes.py --execute")
    if counts.get("recipe.recipe_nutrients") == 0:
        issues.append("recipe.recipe_nutrients is empty — run load_recipe_nutrients.py --execute")

    return issues, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dietary tagging environment")
    parser.add_argument("--skip-db", action="store_true", help="Only check local files")
    parser.add_argument("--local-only", action="store_true", help="Alias for --skip-db")
    args = parser.parse_args()
    if args.local_only:
        args.skip_db = True

    print("=== Dietary tagging environment check ===\n", flush=True)
    print("Local files:")
    local_issues = check_local_files()
    print(flush=True)

    if local_issues:
        print("Local warnings:")
        for msg in local_issues:
            print(f"  - {msg}", flush=True)
        print(flush=True)

    if args.skip_db:
        if local_issues:
            sys.exit(1)
        print("Local checks passed (DB skipped).", flush=True)
        return

    db_issues, counts = check_database()
    if counts:
        print("Table row counts:")
        for table, n in counts.items():
            print(f"  {table}: {n if n is not None else 'N/A'}", flush=True)
        print(flush=True)

    all_issues = local_issues + db_issues
    if all_issues:
        print("Issues:")
        for msg in all_issues:
            print(f"  - {msg}", flush=True)
        sys.exit(1)

    print("All checks passed.", flush=True)


if __name__ == "__main__":
    main()
