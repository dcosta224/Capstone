#!/usr/bin/env python3
"""
Build recipe semantic_text features and load MiniLM embeddings into Supabase.

Ports the embedding pipeline from exploration.ipynb into the `recipe` schema:
  recipe.recipe_nlg_features   — title_clean, semantic_text, ingredient_count
  recipe.recipe_nlg_embedding  — vector(384) keyed by recipe_nlg.id

Prerequisites:
  - recipe.recipe_nlg loaded (`uv run python scripts/load_recipes.py`)
  - pgvector extension enabled on Supabase (Dashboard → Database → Extensions)

Usage:
  uv run python scripts/load_recipe_embeddings.py --schema-only
  uv run python scripts/load_recipe_embeddings.py --limit 1000
  uv run python scripts/load_recipe_embeddings.py
  uv run python scripts/load_recipe_embeddings.py --from-csv
  uv run python scripts/load_recipe_embeddings.py --features-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from recipe_embedding import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    prepare_recipe_features,
    vector_literal,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPE_NLG_CSV = ROOT / "Data" / "recipes" / "RecipeNLG.csv"
SCHEMA_SQL = ROOT / "sql" / "11_create_recipe_embedding_schema.sql"

FEATURES_SQL = """
INSERT INTO recipe.recipe_nlg_features
    (recipe_id, title_clean, semantic_text, ingredient_count)
VALUES %s
ON CONFLICT (recipe_id) DO UPDATE SET
    title_clean = EXCLUDED.title_clean,
    semantic_text = EXCLUDED.semantic_text,
    ingredient_count = EXCLUDED.ingredient_count,
    created_at = now()
"""

EMBEDDING_SQL = """
INSERT INTO recipe.recipe_nlg_embedding
    (recipe_id, model, dims, embedding)
VALUES %s
ON CONFLICT (recipe_id) DO UPDATE SET
    model = EXCLUDED.model,
    dims = EXCLUDED.dims,
    embedding = EXCLUDED.embedding,
    created_at = now()
"""


def configure_session(cur) -> None:
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET lock_timeout = 0")


def apply_schema_sql(cur, path: Path) -> None:
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            cur.execute(stmt)


def fetch_recipes_from_db(conn, limit: int | None) -> pd.DataFrame:
    sql = """
        SELECT id, title, ingredients, directions
        FROM recipe.recipe_nlg
        ORDER BY id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def fetch_recipes_from_csv(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    if "Unnamed: 0" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    return df


def existing_feature_ids(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT recipe_id FROM recipe.recipe_nlg_features")
        return {int(row[0]) for row in cur.fetchall()}


def existing_embedding_ids(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT recipe_id FROM recipe.recipe_nlg_embedding")
        return {int(row[0]) for row in cur.fetchall()}


def load_features(
    conn,
    features: pd.DataFrame,
    batch_size: int,
    skip_ids: set[int],
) -> int:
    rows = [
        (
            int(row.id),
            row.title_clean,
            row.semantic_text,
            int(row.ingredient_count),
        )
        for row in features.itertuples(index=False)
        if int(row.id) not in skip_ids
    ]
    if not rows:
        return 0

    total = 0
    with conn.cursor() as cur:
        configure_session(cur)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            psycopg2.extras.execute_values(
                cur,
                FEATURES_SQL,
                chunk,
                page_size=len(chunk),
            )
            total += len(chunk)
            conn.commit()
            print(f"  features: {total:,} rows committed", flush=True)
    return total


def encode_embeddings(texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL.split("/", 1)[-1])
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_embeddings(
    conn,
    features: pd.DataFrame,
    encode_batch_size: int,
    insert_batch_size: int,
    skip_ids: set[int],
) -> int:
    pending = features[~features["id"].isin(skip_ids)].copy()
    if pending.empty:
        return 0

    texts = pending["semantic_text"].tolist()
    ids = pending["id"].astype(int).tolist()

    print(f"Encoding {len(texts):,} recipes with {EMBEDDING_MODEL} …", flush=True)
    vectors = encode_embeddings(texts, batch_size=encode_batch_size)

    if vectors.shape[1] != EMBEDDING_DIMS:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMS} dims, got {vectors.shape[1]}"
        )

    rows = [
        (recipe_id, EMBEDDING_MODEL, EMBEDDING_DIMS, vector_literal(vectors[i]))
        for i, recipe_id in enumerate(ids)
    ]

    total = 0
    with conn.cursor() as cur:
        configure_session(cur)
        for start in range(0, len(rows), insert_batch_size):
            chunk = rows[start : start + insert_batch_size]
            psycopg2.extras.execute_values(
                cur,
                EMBEDDING_SQL,
                chunk,
                template="(%s, %s, %s, %s::vector)",
                page_size=len(chunk),
            )
            total += len(chunk)
            conn.commit()
            print(f"  embeddings: {total:,} rows committed", flush=True)
    return total


def print_counts(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 'recipe_nlg' AS tbl, COUNT(*)::bigint FROM recipe.recipe_nlg
            UNION ALL
            SELECT 'recipe_nlg_features', COUNT(*)::bigint
            FROM recipe.recipe_nlg_features
            UNION ALL
            SELECT 'recipe_nlg_embedding', COUNT(*)::bigint
            FROM recipe.recipe_nlg_embedding
            """
        )
        for tbl, cnt in cur.fetchall():
            print(f"  recipe.{tbl}: {cnt:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Apply sql/11_create_recipe_embedding_schema.sql and exit",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Read RecipeNLG.csv instead of recipe.recipe_nlg",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N raw recipes (before dedup)",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Load recipe_nlg_features without encoding embeddings",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="SentenceTransformer encode batch size (default 256)",
    )
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=2000,
        help="Rows per INSERT/commit (default 2000)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-upsert rows even if recipe_id already exists",
    )
    args = parser.parse_args()
    load_dotenv()

    t0 = time.perf_counter()
    with connect() as conn:
        with conn.cursor() as cur:
            configure_session(cur)
            print("Applying", SCHEMA_SQL)
            apply_schema_sql(cur, SCHEMA_SQL)
            conn.commit()

        if args.schema_only:
            print("Schema applied.")
            return

        if args.from_csv:
            if not RECIPE_NLG_CSV.is_file():
                sys.exit(f"Missing {RECIPE_NLG_CSV}")
            print(f"Reading {RECIPE_NLG_CSV} …")
            raw = fetch_recipes_from_csv(RECIPE_NLG_CSV, args.limit)
        else:
            print("Reading recipe.recipe_nlg …")
            raw = fetch_recipes_from_db(conn, args.limit)

        print(f"Raw recipes: {len(raw):,}")
        features = prepare_recipe_features(raw)
        print(f"After dedup: {len(features):,}")

        skip_features = set() if args.no_resume else existing_feature_ids(conn)
        skip_embeddings = set() if args.no_resume else existing_embedding_ids(conn)

        print("Loading recipe.recipe_nlg_features …")
        n_features = load_features(
            conn,
            features,
            batch_size=args.insert_batch_size,
            skip_ids=skip_features,
        )
        print(f"  features loaded this run: {n_features:,}")

        if not args.features_only:
            print("Loading recipe.recipe_nlg_embedding …")
            n_embeddings = load_embeddings(
                conn,
                features,
                encode_batch_size=args.encode_batch_size,
                insert_batch_size=args.insert_batch_size,
                skip_ids=skip_embeddings,
            )
            print(f"  embeddings loaded this run: {n_embeddings:,}")

        print("\nTable counts:")
        print_counts(conn)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
