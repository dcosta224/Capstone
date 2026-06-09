#!/usr/bin/env python3
"""Load MVP fully-resolved recipes into recipe.resolved_recipes from v4 pipeline artifacts.

Usage:
  uv run python scripts/load_resolved_recipes.py --dry-run
  uv run python scripts/load_resolved_recipes.py --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from resolved_recipe_portion import (
    DEFAULT_V4_DIR,
    build_portion_label,
    extract_portion_id,
    fully_resolved_recipe_ids,
    load_v4_matches,
    normalize_fdc_id,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPE_CSV = ROOT / "Data" / "recipes" / "RecipeNLG.csv"
SCHEMA_SQL = ROOT / "sql" / "11_create_resolved_recipes.sql"

INSERT_COLS = [
    "recipe_id",
    "ingredient_idx",
    "recipe_name",
    "ingredient",
    "fdc_id",
    "fdc_description",
    "portion_id",
    "portion_label",
    "quantity",
    "unit",
    "gram_weight",
    "amount_kind",
    "grams_status",
    "feasibility_version",
]


def apply_schema_sql(cur, path: Path) -> None:
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    sql = "\n".join(lines)
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)


def load_titles_from_db(cur, recipe_ids: list[int]) -> dict[int, str]:
    if not recipe_ids:
        return {}
    cur.execute(
        "SELECT id, title FROM recipe.recipe_nlg WHERE id = ANY(%s)",
        (recipe_ids,),
    )
    return {int(r[0]): str(r[1]) for r in cur.fetchall()}


def load_titles_from_csv(recipe_ids: set[int]) -> dict[int, str]:
    if not RECIPE_CSV.is_file():
        return {}
    id_set = recipe_ids
    out: dict[int, str] = {}
    for chunk in pd.read_csv(RECIPE_CSV, chunksize=200_000):
        id_col = chunk.columns[0]
        title_col = "title"
        sel = chunk[chunk[id_col].astype(int).isin(id_set)]
        for _, r in sel.iterrows():
            out[int(r[id_col])] = str(r[title_col])
    return out


def fetch_fdc_descriptions(cur, fdc_ids: list[int]) -> dict[int, str]:
    if not fdc_ids:
        return {}
    rows: list = []
    try:
        cur.execute(
            "SELECT fdc_id, description FROM usda.food_4macro WHERE fdc_id = ANY(%s)",
            (fdc_ids,),
        )
        rows = cur.fetchall()
    except Exception:
        cur.connection.rollback()
    if not rows:
        cur.execute(
            "SELECT fdc_id, description FROM usda.food WHERE fdc_id = ANY(%s)",
            (fdc_ids,),
        )
        rows = cur.fetchall()
        print("  note: used usda.food for fdc descriptions", flush=True)
    return {int(r[0]): str(r[1]) for r in rows if r[1]}


def fetch_portion_labels(cur, portion_ids: list[int]) -> dict[int, str]:
    if not portion_ids:
        return {}
    cur.execute(
        """
        SELECT fp.id, fp.portion_description, fp.modifier, mu.name AS measure_unit_name
        FROM usda.food_portion fp
        LEFT JOIN usda.measure_unit mu ON fp.measure_unit_id = mu.id
        WHERE fp.id = ANY(%s)
        """,
        (portion_ids,),
    )
    return {
        int(r[0]): build_portion_label(r[1], r[2], r[3])
        for r in cur.fetchall()
    }


def build_resolved_rows(
    matches: pd.DataFrame,
    recipe_ids: set[int],
    titles: dict[int, str],
    fdc_desc: dict[int, str],
    portion_labels: dict[int, str],
    feasibility_version: int | None,
) -> list[tuple]:
    sub = matches[matches["recipe_id"].astype(int).isin(recipe_ids)].copy()
    rows: list[tuple] = []
    for r in sub.itertuples(index=False):
        rid = int(r.recipe_id)
        iidx = int(r.ingredient_idx)
        fdc = normalize_fdc_id(getattr(r, "llm_fdc_id", None))
        amount_kind = str(
            getattr(r, "amount_kind_final", None)
            or getattr(r, "amount_kind", None)
            or ""
        )
        pid = None
        if amount_kind not in ("mass",):
            pid = extract_portion_id(r)
        qty = getattr(r, "quantity", None)
        qty = float(qty) if qty is not None and pd.notna(qty) else None
        unit = getattr(r, "unit", None)
        unit = str(unit) if unit is not None and pd.notna(unit) else None
        rows.append(
            (
                rid,
                iidx,
                titles.get(rid),
                str(r.ingredient),
                fdc,
                fdc_desc.get(fdc) if fdc is not None else None,
                pid,
                portion_labels.get(pid) if pid is not None else None,
                qty,
                unit,
                float(r.grams),
                amount_kind or None,
                str(getattr(r, "grams_status", None) or "") or None,
                feasibility_version,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load recipe.resolved_recipes (MVP 106)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_V4_DIR)
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        parser.error("Specify --dry-run or --execute")

    matches, manifest = load_v4_matches(args.artifacts_dir)
    recipe_ids = fully_resolved_recipe_ids(matches)
    feasibility_version = manifest.get("feasibility_version")

    print(f"Matches: {len(matches):,} lines", flush=True)
    print(f"Fully resolved recipes: {len(recipe_ids)}", flush=True)

    sub = matches[matches["recipe_id"].astype(int).isin(recipe_ids)]
    print(f"Ingredient lines to load: {len(sub):,}", flush=True)

    load_dotenv()
    conn = connect()
    try:
        with conn.cursor() as cur:
            titles = load_titles_from_db(cur, sorted(recipe_ids))
        missing = recipe_ids - set(titles.keys())
        if missing:
            titles.update(load_titles_from_csv(missing))
            print(f"  titles from CSV fallback: {len(missing)} recipes", flush=True)

        fdc_ids = sorted(
            {
                normalize_fdc_id(x)
                for x in sub["llm_fdc_id"]
                if normalize_fdc_id(x) is not None
            }
        )
        portion_ids: list[int] = []
        for r in sub.itertuples(index=False):
            ak = str(getattr(r, "amount_kind_final", "") or "")
            if ak != "mass":
                pid = extract_portion_id(r)
                if pid is not None:
                    portion_ids.append(pid)
        portion_ids = sorted(set(portion_ids))

        with conn.cursor() as cur:
            fdc_desc = fetch_fdc_descriptions(cur, fdc_ids)
            portion_labels = fetch_portion_labels(cur, portion_ids)

        rows = build_resolved_rows(
            matches,
            recipe_ids,
            titles,
            fdc_desc,
            portion_labels,
            feasibility_version,
        )

        n_with_portion = sum(1 for r in rows if r[6] is not None)
        n_mass = sum(1 for r in rows if r[11] == "mass")
        print(f"Rows built: {len(rows)}", flush=True)
        print(f"  with portion_id: {n_with_portion}", flush=True)
        print(f"  mass lines: {n_mass}", flush=True)
        print(f"  fdc descriptions found: {len(fdc_desc)}/{len(fdc_ids)}", flush=True)

        if args.dry_run:
            print("\nSample rows (first 3):", flush=True)
            for row in rows[:3]:
                print(f"  {row[:8]}...", flush=True)
            return

        with conn.cursor() as cur:
            apply_schema_sql(cur, SCHEMA_SQL)
            cur.execute("TRUNCATE recipe.resolved_recipes")
            sql = (
                f"INSERT INTO recipe.resolved_recipes ({', '.join(INSERT_COLS)}) VALUES %s"
            )
            psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
        conn.commit()
        print(f"Loaded {len(rows)} rows into recipe.resolved_recipes", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
