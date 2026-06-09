#!/usr/bin/env python3
"""
Build recipe semantic_text features and load MiniLM embeddings into Supabase.

Reads from local RecipeNLG.csv in chunks (default) and uploads to:
  recipe.recipe_nlg_features   — title_clean, semantic_text, ingredient_count
  recipe.recipe_nlg_embedding  — vector(384) keyed by recipe_nlg.id

Only rows whose id exists in recipe.recipe_nlg are uploaded (skips ids not loaded yet).
By default resumes automatically: queries Supabase once at startup for ids already
in recipe_nlg_features / recipe_nlg_embedding and skips them (use --no-resume to redo).

Prerequisites:
  - recipe.recipe_nlg populated (full or partial load via load_recipes.py)
  - pgvector extension enabled on Supabase (Dashboard → Database → Extensions)

Usage:
  uv run python scripts/load_recipe_embeddings.py --schema-only
  uv run python scripts/load_recipe_embeddings.py --limit 5000
  uv run python scripts/load_recipe_embeddings.py
  uv run python scripts/load_recipe_embeddings.py --features-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import psycopg2.extras
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from recipe_embedding import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    build_recipe_features,
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

_MODEL = None


def configure_session(cur) -> None:
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET lock_timeout = 0")


def apply_schema_sql(cur, path: Path) -> None:
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            cur.execute(stmt)


def normalize_csv_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in chunk.columns and "id" not in chunk.columns:
        chunk = chunk.rename(columns={"Unnamed: 0": "id"})
    return chunk


def count_csv_data_rows(path: Path) -> int:
    """Fast newline count minus header."""
    lines = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            lines += block.count(b"\n")
    return max(lines - 1, 0)


def iter_csv_chunks(
    path: Path,
    chunk_size: int,
    limit: int | None,
) -> Iterator[pd.DataFrame]:
    rows_read = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk = normalize_csv_chunk(chunk)
        if limit is not None:
            remaining = limit - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)
        rows_read += len(chunk)
        yield chunk
        if limit is not None and rows_read >= limit:
            break


@dataclass
class ResumeState:
    recipe_nlg_ids: set[int]
    feature_ids: set[int]
    embedding_ids: set[int]


def load_id_set(conn, query: str, label: str) -> set[int]:
    tqdm.write(f"Querying Supabase: {label} …")
    with conn.cursor() as cur:
        configure_session(cur)
        cur.execute(query)
        rows = cur.fetchall()
    ids = {int(row[0]) for row in rows}
    tqdm.write(f"  {label}: {len(ids):,}")
    return ids


def fetch_resume_state(conn, *, features_only: bool) -> ResumeState:
    """Load all ids already present in Supabase (one query per table)."""
    recipe_nlg_ids = load_id_set(
        conn,
        "SELECT id FROM recipe.recipe_nlg",
        "recipe.recipe_nlg ids",
    )
    feature_ids = load_id_set(
        conn,
        "SELECT recipe_id FROM recipe.recipe_nlg_features",
        "recipe.recipe_nlg_features ids",
    )
    if features_only:
        return ResumeState(recipe_nlg_ids, feature_ids, set())

    embedding_ids = load_id_set(
        conn,
        "SELECT recipe_id FROM recipe.recipe_nlg_embedding",
        "recipe.recipe_nlg_embedding ids",
    )
    return ResumeState(recipe_nlg_ids, feature_ids, embedding_ids)


def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBEDDING_MODEL.split("/", 1)[-1])
    return _MODEL


def encode_texts(texts: list[str], batch_size: int) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_feature_rows(conn, rows: list[tuple], batch_size: int) -> int:
    total = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        with conn.cursor() as cur:
            configure_session(cur)
            psycopg2.extras.execute_values(
                cur,
                FEATURES_SQL,
                chunk,
                page_size=len(chunk),
            )
        conn.commit()
        total += len(chunk)
    return total


def load_embedding_rows(conn, rows: list[tuple], batch_size: int) -> int:
    total = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        with conn.cursor() as cur:
            configure_session(cur)
            psycopg2.extras.execute_values(
                cur,
                EMBEDDING_SQL,
                chunk,
                template="(%s, %s, %s, %s::vector)",
                page_size=len(chunk),
            )
        conn.commit()
        total += len(chunk)
    return total


def process_chunk(
    conn,
    raw_chunk: pd.DataFrame,
    *,
    insert_batch_size: int,
    encode_batch_size: int,
    features_only: bool,
    resume: ResumeState | None,
) -> tuple[int, int, int]:
    """Returns (n_features, n_embeddings, n_skipped_already_done)."""
    raw_chunk = normalize_csv_chunk(raw_chunk)
    if "id" not in raw_chunk.columns:
        raise ValueError("CSV chunk missing id column")

    chunk_ids = raw_chunk["id"].astype(int).tolist()
    skipped_before = 0
    if resume is not None:
        in_db = {i for i in chunk_ids if i in resume.recipe_nlg_ids}
        skip_features = resume.feature_ids
        skip_embeddings = resume.embedding_ids
        if features_only:
            skipped_before = sum(1 for i in in_db if i in skip_features)
        else:
            skipped_before = sum(1 for i in in_db if i in skip_embeddings)
    else:
        with conn.cursor() as cur:
            configure_session(cur)
            cur.execute(
                "SELECT id FROM recipe.recipe_nlg WHERE id = ANY(%s)",
                (chunk_ids,),
            )
            in_db = {int(row[0]) for row in cur.fetchall()}
        skip_features = set()
        skip_embeddings = set()

    if not in_db:
        return 0, 0, 0

    if resume is not None:
        if features_only:
            if skipped_before == len(in_db):
                return 0, 0, skipped_before
        else:
            needs_embeddings = [i for i in in_db if i not in skip_embeddings]
            needs_features = [i for i in in_db if i not in skip_features]
            if not needs_embeddings and not needs_features:
                return 0, 0, skipped_before

    features = build_recipe_features(raw_chunk, dedup=False)
    features = features[features["id"].isin(in_db)].copy()

    feature_rows = [
        (int(row.id), row.title_clean, row.semantic_text, int(row.ingredient_count))
        for row in features.itertuples(index=False)
        if int(row.id) not in skip_features
    ]
    n_features = load_feature_rows(conn, feature_rows, insert_batch_size)

    n_embeddings = 0
    if not features_only:
        embed_df = features[~features["id"].isin(skip_embeddings)]
        if not embed_df.empty:
            vectors = encode_texts(
                embed_df["semantic_text"].tolist(),
                batch_size=encode_batch_size,
            )
            if vectors.shape[1] != EMBEDDING_DIMS:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIMS} dims, got {vectors.shape[1]}"
                )
            embedding_rows = [
                (
                    int(recipe_id),
                    EMBEDDING_MODEL,
                    EMBEDDING_DIMS,
                    vector_literal(vectors[i]),
                )
                for i, recipe_id in enumerate(embed_df["id"].astype(int).tolist())
            ]
            n_embeddings = load_embedding_rows(
                conn, embedding_rows, insert_batch_size
            )
            if resume is not None:
                resume.embedding_ids.update(embed_df["id"].astype(int).tolist())

    if resume is not None and feature_rows:
        resume.feature_ids.update(int(row[0]) for row in feature_rows)

    return n_features, n_embeddings, skipped_before


def print_embedding_counts(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
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
        "--csv",
        type=Path,
        default=RECIPE_NLG_CSV,
        help=f"Local RecipeNLG CSV (default: {RECIPE_NLG_CSV})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N raw CSV rows",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="CSV rows per read/encode/upload cycle (default 50000)",
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

    if not args.schema_only and not args.csv.is_file():
        sys.exit(f"Missing {args.csv}")

    t0 = time.perf_counter()
    total_features = 0
    total_embeddings = 0
    total_rows = 0
    total_skipped = 0

    with connect() as conn:
        with conn.cursor() as cur:
            configure_session(cur)
            print("Applying", SCHEMA_SQL)
            apply_schema_sql(cur, SCHEMA_SQL)
            conn.commit()

        if args.schema_only:
            print("Schema applied.")
            return

        file_rows = count_csv_data_rows(args.csv)
        target_rows = min(file_rows, args.limit) if args.limit else file_rows
        tqdm.write(
            f"CSV rows: {file_rows:,}"
            + (f" (processing first {target_rows:,})" if args.limit else "")
        )
        tqdm.write(f"Chunk size: {args.chunk_size:,}")

        resume: ResumeState | None = None
        if not args.no_resume:
            resume = fetch_resume_state(conn, features_only=args.features_only)
            if args.features_only:
                tqdm.write(
                    f"Resume: {len(resume.feature_ids):,} features already in Supabase"
                )
            else:
                tqdm.write(
                    f"Resume: {len(resume.embedding_ids):,} embeddings, "
                    f"{len(resume.feature_ids):,} features already in Supabase"
                )
        else:
            tqdm.write("Resume disabled (--no-resume); all matching rows will be upserted")

        tqdm.write(f"Loading {EMBEDDING_MODEL} …")
        get_embedding_model()

        progress = tqdm(
            total=target_rows,
            unit="rows",
            desc="embed → supabase",
            dynamic_ncols=True,
            smoothing=0.03,
        )
        try:
            for raw_chunk in iter_csv_chunks(
                args.csv, args.chunk_size, args.limit
            ):
                n_features, n_embeddings, n_skipped = process_chunk(
                    conn,
                    raw_chunk,
                    insert_batch_size=args.insert_batch_size,
                    encode_batch_size=args.encode_batch_size,
                    features_only=args.features_only,
                    resume=resume,
                )
                total_features += n_features
                total_embeddings += n_embeddings
                total_skipped += n_skipped
                total_rows += len(raw_chunk)
                progress.update(len(raw_chunk))
                progress.set_postfix(
                    features=f"{total_features:,}",
                    embeddings=f"{total_embeddings:,}",
                    skipped=f"{total_skipped:,}",
                    refresh=True,
                )
        finally:
            progress.close()

        tqdm.write("\nEmbedding table counts:")
        print_embedding_counts(conn)

    elapsed = time.perf_counter() - t0
    tqdm.write(
        f"Done in {elapsed:.1f}s — "
        f"features: {total_features:,}, embeddings: {total_embeddings:,}, "
        f"skipped (already done): {total_skipped:,}"
    )


if __name__ == "__main__":
    main()
