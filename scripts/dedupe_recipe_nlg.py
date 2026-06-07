#!/usr/bin/env python3
"""
Deduplicate recipe.recipe_nlg using Kadin's hybrid pipeline from exploration.ipynb.

Phase 1 — exact duplicates: same title + ingredients + directions (keep lowest id).
Phase 2 — hybrid semantic duplicates:
  MiniLM embeddings → FAISS k-NN → hybrid score (semantic + ingredient Jaccard + title)
  → connected components → keep richest recipe per cluster (ingredient_count, then directions length).

By default this is a dry run. Pass --execute to DELETE rows from recipe.recipe_nlg
(cascades to recipe_nlg_features / recipe_nlg_embedding).

Usage:
  uv run python scripts/dedupe_recipe_nlg.py --dry-run
  uv run python scripts/dedupe_recipe_nlg.py --execute
  uv run python scripts/dedupe_recipe_nlg.py --limit 10000 --dry-run
  uv run python scripts/dedupe_recipe_nlg.py --skip-exact --execute
  uv run python scripts/dedupe_recipe_nlg.py --use-db-embeddings --execute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from recipe_dedup import (
    DedupConfig,
    encode_semantic_texts,
    exact_duplicate_ids,
    parse_pgvector,
    plan_hybrid_dedup,
    prepare_recipe_clean_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "dedup"


def configure_session(cur) -> None:
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET lock_timeout = 0")


def fetch_recipes(conn) -> pd.DataFrame:
    sql = """
        SELECT id, title, ingredients, directions
        FROM recipe.recipe_nlg
        ORDER BY id
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def fetch_embeddings_from_db(conn) -> pd.DataFrame:
    sql = """
        SELECT e.recipe_id, e.embedding::text AS embedding
        FROM recipe.recipe_nlg_embedding e
        INNER JOIN recipe.recipe_nlg r ON r.id = e.recipe_id
        ORDER BY e.recipe_id
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def delete_ids_batched(conn, ids: list[int], batch_size: int) -> int:
    if not ids:
        return 0
    deleted = 0
    sql = "DELETE FROM recipe.recipe_nlg WHERE id = ANY(%s)"
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        with conn.cursor() as cur:
            configure_session(cur)
            cur.execute(sql, (chunk,))
        conn.commit()
        deleted += len(chunk)
        print(f"  deleted: {deleted:,}/{len(ids):,}", flush=True)
    return deleted


def exact_delete_sql(conn, batch_size: int) -> int:
    """Delete exact duplicates in SQL (keeps lowest id per content key)."""
    count_sql = """
        SELECT COUNT(*)::bigint
        FROM recipe.recipe_nlg a
        INNER JOIN recipe.recipe_nlg b
            ON a.title = b.title
           AND a.ingredients = b.ingredients
           AND a.directions = b.directions
           AND a.id > b.id
    """
    with conn.cursor() as cur:
        cur.execute(count_sql)
        to_delete = int(cur.fetchone()[0])
    if to_delete == 0:
        return 0

    print(f"  exact duplicates to delete: {to_delete:,}", flush=True)
    total = 0
    while True:
        with conn.cursor() as cur:
            configure_session(cur)
            cur.execute(
                """
                WITH doomed AS (
                    SELECT a.id
                    FROM recipe.recipe_nlg a
                    INNER JOIN recipe.recipe_nlg b
                        ON a.title = b.title
                       AND a.ingredients = b.ingredients
                       AND a.directions = b.directions
                       AND a.id > b.id
                    ORDER BY a.id
                    LIMIT %s
                )
                DELETE FROM recipe.recipe_nlg
                WHERE id IN (SELECT id FROM doomed)
                RETURNING id
                """,
                (batch_size,),
            )
            removed = cur.rowcount
        conn.commit()
        if removed == 0:
            break
        total += removed
        print(f"  exact deleted: {total:,}/{to_delete:,}", flush=True)
    return total


def write_artifacts(
    output_dir: Path,
    *,
    keeper_ids: list[int],
    exact_delete_ids: list[int],
    hybrid_delete_ids: list[int],
    summary: dict,
    validated_pairs: pd.DataFrame | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": keeper_ids}).to_csv(output_dir / "keeper_ids.csv", index=False)
    pd.DataFrame({"id": exact_delete_ids}).to_csv(
        output_dir / "exact_delete_ids.csv", index=False
    )
    pd.DataFrame({"id": hybrid_delete_ids}).to_csv(
        output_dir / "hybrid_delete_ids.csv", index=False
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if validated_pairs is not None and not validated_pairs.empty:
        validated_pairs.head(10_000).to_csv(
            output_dir / "validated_pairs_sample.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute delete plan and write artifacts; do not modify the database",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Apply exact + hybrid deletions to recipe.recipe_nlg",
    )
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="Skip exact duplicate removal (phase 1)",
    )
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip hybrid semantic deduplication (phase 2)",
    )
    parser.add_argument(
        "--use-db-embeddings",
        action="store_true",
        help="Use recipe.recipe_nlg_embedding instead of encoding in memory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N recipes (testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Write keeper/delete manifests here (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--semantic-threshold", type=float, default=0.95)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--delete-batch-size", type=int, default=10_000)
    args = parser.parse_args()
    load_dotenv()

    cfg = DedupConfig(
        k=args.k,
        semantic_pair_threshold=args.semantic_threshold,
        encode_batch_size=args.encode_batch_size,
    )

    t0 = time.perf_counter()
    with connect() as conn:
        print("Loading recipe.recipe_nlg …")
        raw = fetch_recipes(conn)
        if args.limit is not None:
            raw = raw.head(args.limit).copy()
        print(f"  rows: {len(raw):,}")

        clean = prepare_recipe_clean_frame(raw)
        exact_ids = [] if args.skip_exact else exact_duplicate_ids(clean)
        print(f"Exact duplicate ids (phase 1): {len(exact_ids):,}")

        if args.execute and not args.skip_exact and exact_ids:
            print("Deleting exact duplicates …")
            exact_delete_sql(conn, args.delete_batch_size)
            raw = fetch_recipes(conn)
            if args.limit is not None:
                raw = raw.head(args.limit).copy()
            clean = prepare_recipe_clean_frame(raw)

        hybrid_delete_ids: list[int] = []
        keeper_ids: list[int] = sorted(clean["id"].astype(int).tolist())
        validated_pairs = pd.DataFrame()

        hybrid_clean = clean[~clean["id"].isin(exact_ids)].copy()

        if not args.skip_hybrid:
            if args.use_db_embeddings:
                print("Loading embeddings from recipe.recipe_nlg_embedding …")
                emb_df = fetch_embeddings_from_db(conn)
                if args.limit is not None:
                    emb_df = emb_df[emb_df["recipe_id"].isin(hybrid_clean["id"])]
                merged = hybrid_clean.loc[:, ["id", "semantic_text"]].merge(
                    emb_df, left_on="id", right_on="recipe_id", how="inner"
                )
                if len(merged) != len(hybrid_clean):
                    missing = len(hybrid_clean) - len(merged)
                    sys.exit(
                        f"--use-db-embeddings requested but {missing:,} recipes "
                        "lack embeddings. Run load_recipe_embeddings.py first or "
                        "omit --use-db-embeddings."
                    )
                vectors = np.vstack(
                    merged["embedding"].map(parse_pgvector).to_numpy()
                )
                work_clean = clean.set_index("id").loc[merged["id"]].reset_index()
            else:
                print("Encoding semantic_text with MiniLM …")
                vectors = encode_semantic_texts(
                    hybrid_clean["semantic_text"].tolist(),
                    batch_size=cfg.encode_batch_size,
                )
                work_clean = hybrid_clean

            print("Planning hybrid deduplication …")
            hybrid_delete_set, _hybrid_pairs, validated_pairs = plan_hybrid_dedup(
                work_clean, vectors, cfg
            )
            hybrid_delete_ids = sorted(hybrid_delete_set)
            hybrid_keepers = set(work_clean["id"].astype(int)) - hybrid_delete_set
            keeper_ids = sorted(hybrid_keepers)
            print(f"Hybrid duplicate ids (phase 2): {len(hybrid_delete_ids):,}")
            print(f"Keepers after hybrid: {len(keeper_ids):,}")

            if args.execute and hybrid_delete_ids:
                print("Deleting hybrid duplicates …")
                delete_ids_batched(conn, hybrid_delete_ids, args.delete_batch_size)

        final_keepers = sorted(
            set(keeper_ids) if not args.skip_hybrid else set(clean["id"].astype(int)) - set(exact_ids)
        )
        summary = {
            "mode": "execute" if args.execute else "dry-run",
            "input_rows": int(len(raw)),
            "exact_delete_count": len(exact_ids),
            "hybrid_delete_count": len(hybrid_delete_ids),
            "keeper_count": len(final_keepers),
            "validated_pair_count": int(len(validated_pairs)),
            "config": {
                "k": cfg.k,
                "semantic_pair_threshold": cfg.semantic_pair_threshold,
                "validated_semantic": cfg.validated_semantic,
                "validated_jaccard": cfg.validated_jaccard,
                "validated_hybrid": cfg.validated_hybrid,
                "use_db_embeddings": args.use_db_embeddings,
                "skip_exact": args.skip_exact,
                "skip_hybrid": args.skip_hybrid,
            },
        }
        write_artifacts(
            args.output_dir,
            keeper_ids=final_keepers,
            exact_delete_ids=exact_ids,
            hybrid_delete_ids=hybrid_delete_ids,
            summary=summary,
            validated_pairs=validated_pairs,
        )

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*)::bigint FROM recipe.recipe_nlg")
            remaining = int(cur.fetchone()[0])

    elapsed = time.perf_counter() - t0
    print(f"\nArtifacts: {args.output_dir}")
    print(json.dumps(summary, indent=2))
    print(f"recipe.recipe_nlg rows now: {remaining:,}")
    print(f"Done in {elapsed:.1f}s")
    if args.dry_run:
        print("\nDry run only. Re-run with --execute to delete duplicates.")


if __name__ == "__main__":
    main()
