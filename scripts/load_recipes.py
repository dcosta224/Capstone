#!/usr/bin/env python3
"""
Create the `recipe` schema and load Open Recipes + RecipeNLG into Supabase.

Usage:
  pip install -r requirements.txt
  python scripts/load_recipes.py
  python scripts/load_recipes.py --skip-nlg
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
COPY_SQL = (
    "COPY recipe.recipe_nlg (id, title, ingredients, directions, link, source, ner) "
    "FROM STDIN WITH (FORMAT csv, HEADER true, QUOTE '\"', ESCAPE '\"')"
)


def configure_session(cur) -> None:
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET lock_timeout = 0")


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
    if not raw:
        return "[]"
    json.loads(raw)
    return raw


def apply_schema_sql(cur, path: Path) -> None:
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            cur.execute(stmt)


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


def nlg_row_tuple(row: list[str]) -> list:
    idx, title, ingredients, directions, link, source, ner = row
    return [
        idx,
        title,
        parse_json_field(ingredients),
        parse_json_field(directions),
        link,
        source,
        parse_json_field(ner),
    ]


def copy_chunk(cur, rows: list[list]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", newline="", suffix=".csv"
    ) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(NLG_COLUMNS)
        writer.writerows(rows)
        tmp.flush()
        tmp.seek(0)
        cur.copy_expert(COPY_SQL, tmp)


def load_recipe_nlg_chunked(
    conn, path: Path, chunk_size: int = 75_000
) -> int:
    """COPY in chunks with a commit after each chunk (avoids Supabase timeouts)."""
    total = 0
    chunk: list[list] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[1:] == ["title", "ingredients", "directions", "link", "source", "NER"]

        for row in reader:
            chunk.append(nlg_row_tuple(row))
            if len(chunk) >= chunk_size:
                with conn.cursor() as cur:
                    configure_session(cur)
                    copy_chunk(cur, chunk)
                conn.commit()
                total += len(chunk)
                chunk.clear()
                print(f"  recipe_nlg: {total:,} rows committed", flush=True)

        if chunk:
            with conn.cursor() as cur:
                configure_session(cur)
                copy_chunk(cur, chunk)
            conn.commit()
            total += len(chunk)

    print(f"  recipe_nlg: {total:,} rows loaded")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nlg", action="store_true")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=75_000,
        help="Rows per RecipeNLG COPY/commit (default 75000)",
    )
    args = parser.parse_args()
    load_dotenv()

    if not OPEN_RECIPES.is_file():
        sys.exit(f"Missing {OPEN_RECIPES}")
    if not args.skip_nlg and not RECIPE_NLG.is_file():
        sys.exit(f"Missing {RECIPE_NLG}")

    t0 = time.perf_counter()
    with connect() as conn:
        with conn.cursor() as cur:
            configure_session(cur)
            print("Applying", SCHEMA_SQL)
            apply_schema_sql(cur, SCHEMA_SQL)
            conn.commit()

            print("Loading open_recipes.json …")
            load_open_recipes(cur, OPEN_RECIPES)
            conn.commit()

        if not args.skip_nlg:
            print("Loading RecipeNLG.csv (chunked COPY) …")
            load_recipe_nlg_chunked(conn, RECIPE_NLG, chunk_size=args.chunk_size)

        with conn.cursor() as cur:
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
