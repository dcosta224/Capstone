#!/usr/bin/env python3
"""
Create the `recipe` schema and load Open Recipes + RecipeNLG into Supabase.

Usage:
  pip install -r requirements.txt
  python scripts/load_recipes.py
  python scripts/load_recipes.py --skip-nlg   # open_recipes only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OPEN_RECIPES = ROOT / "Data" / "recipes" / "open_recipes.json"
RECIPE_NLG = ROOT / "Data" / "recipes" / "RecipeNLG.csv"
SCHEMA_SQL = ROOT / "sql" / "10_create_recipe_schema.sql"

OPEN_COLUMNS = [
    "id",
    "name",
    "ingredients",
    "url",
    "image",
    "source",
    "description",
    "creator",
    "recipe_yield",
    "recipe_category",
    "cook_time",
    "prep_time",
    "total_time",
    "date_published",
    "date_modified",
    "ts_millis",
]

NLG_COLUMNS = ["id", "title", "ingredients", "directions", "link", "source", "ner"]


def oid(doc: dict) -> str:
    raw = doc.get("_id")
    if isinstance(raw, dict) and "$oid" in raw:
        return str(raw["$oid"])
    return str(raw) if raw is not None else ""


def ts_millis(doc: dict) -> int | None:
    ts = doc.get("ts")
    if isinstance(ts, dict) and "$date" in ts:
        return int(ts["$date"])
    return None


def open_recipe_row(doc: dict) -> tuple:
    return (
        oid(doc),
        doc.get("name"),
        doc.get("ingredients"),
        doc.get("url"),
        doc.get("image"),
        doc.get("source"),
        doc.get("description"),
        doc.get("creator"),
        doc.get("recipeYield"),
        doc.get("recipeCategory"),
        doc.get("cookTime"),
        doc.get("prepTime"),
        doc.get("totalTime"),
        doc.get("datePublished"),
        doc.get("dateModified"),
        ts_millis(doc),
    )


def parse_json_field(raw: str) -> str:
    """Return JSON text for jsonb columns (already valid JSON in CSV)."""
    if not raw:
        return "[]"
    json.loads(raw)
    return raw


def load_open_recipes(cur, path: Path, batch_size: int = 5000) -> int:
    buf: list[tuple] = []
    count = 0
    sql = (
        f"INSERT INTO recipe.open_recipe ({', '.join(OPEN_COLUMNS)}) VALUES %s "
        "ON CONFLICT (id) DO NOTHING"
    )
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            buf.append(open_recipe_row(doc))
            if len(buf) >= batch_size:
                psycopg2.extras.execute_values(cur, sql, buf, page_size=batch_size)
                count += len(buf)
                buf.clear()
                print(f"  open_recipe: {count:,} rows", end="\r", flush=True)
        if buf:
            psycopg2.extras.execute_values(cur, sql, buf, page_size=len(buf))
            count += len(buf)
    print(f"  open_recipe: {count:,} rows loaded")
    return count


def apply_schema_sql(cur, path: Path) -> None:
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            cur.execute(stmt)


def copy_recipe_nlg(cur, path: Path) -> int:
    """Stage RecipeNLG to a temp CSV, then COPY (~2.2M rows)."""
    copy_sql = (
        "COPY recipe.recipe_nlg (id, title, ingredients, directions, link, source, ner) "
        "FROM STDIN WITH (FORMAT csv, HEADER true, QUOTE '\"', ESCAPE '\"')"
    )
    count = 0
    with path.open(newline="", encoding="utf-8") as src, tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", newline="", suffix=".csv"
    ) as tmp:
        reader = csv.reader(src)
        header = next(reader)
        assert header[1:] == ["title", "ingredients", "directions", "link", "source", "NER"]

        writer = csv.writer(tmp)
        writer.writerow(NLG_COLUMNS)
        for row in reader:
            idx, title, ingredients, directions, link, source, ner = row
            writer.writerow(
                [
                    idx,
                    title,
                    parse_json_field(ingredients),
                    parse_json_field(directions),
                    link,
                    source,
                    parse_json_field(ner),
                ]
            )
            count += 1
            if count % 200_000 == 0:
                print(f"  recipe_nlg: staged {count:,} rows", end="\r", flush=True)

        tmp.flush()
        tmp.seek(0)
        cur.copy_expert(copy_sql, tmp)
    print(f"  recipe_nlg: {count:,} rows loaded")
    return count


def load_recipe_nlg_batched(cur, path: Path, batch_size: int = 10_000) -> int:
    """Fallback batched insert if COPY hits memory limits."""
    sql = (
        f"INSERT INTO recipe.recipe_nlg ({', '.join(NLG_COLUMNS)}) VALUES %s "
        "ON CONFLICT (id) DO NOTHING"
    )
    buf: list[tuple] = []
    count = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            idx, title, ingredients, directions, link, source, ner = row
            buf.append(
                (
                    int(idx),
                    title,
                    parse_json_field(ingredients),
                    parse_json_field(directions),
                    link,
                    source,
                    parse_json_field(ner),
                )
            )
            if len(buf) >= batch_size:
                psycopg2.extras.execute_values(
                    cur,
                    sql,
                    buf,
                    template="(%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb)",
                    page_size=batch_size,
                )
                count += len(buf)
                buf.clear()
                print(f"  recipe_nlg: {count:,} rows", end="\r", flush=True)
        if buf:
            psycopg2.extras.execute_values(
                cur,
                sql,
                buf,
                template="(%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb)",
                page_size=len(buf),
            )
            count += len(buf)
    print(f"  recipe_nlg: {count:,} rows loaded")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nlg", action="store_true")
    parser.add_argument(
        "--nlg-batch",
        action="store_true",
        help="Use batched INSERT instead of COPY for RecipeNLG",
    )
    args = parser.parse_args()
    load_dotenv()

    if not OPEN_RECIPES.is_file():
        sys.exit(f"Missing {OPEN_RECIPES}")
    if not args.skip_nlg and not RECIPE_NLG.is_file():
        sys.exit(f"Missing {RECIPE_NLG}")

    t0 = time.perf_counter()
    with connect() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            print("Applying", SCHEMA_SQL)
            apply_schema_sql(cur, SCHEMA_SQL)

            print("Loading open_recipes.json …")
            n_open = load_open_recipes(cur, OPEN_RECIPES)

            n_nlg = 0
            if not args.skip_nlg:
                print("Loading RecipeNLG.csv …")
                if args.nlg_batch:
                    n_nlg = load_recipe_nlg_batched(cur, RECIPE_NLG)
                else:
                    n_nlg = copy_recipe_nlg(cur, RECIPE_NLG)

            conn.commit()
            cur.execute(
                """
                SELECT 'open_recipe' AS tbl, COUNT(*)::bigint FROM recipe.open_recipe
                UNION ALL
                SELECT 'recipe_nlg', COUNT(*)::bigint FROM recipe.recipe_nlg
                """
            )
            counts = cur.fetchall()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    for tbl, cnt in counts:
        print(f"  recipe.{tbl}: {cnt:,}")


if __name__ == "__main__":
    main()
