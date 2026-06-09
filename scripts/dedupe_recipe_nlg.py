#!/usr/bin/env python3
"""
Deduplicate recipe.recipe_nlg using Kadin's hybrid pipeline from exploration.ipynb.

Phase 1 — exact duplicates: same title + ingredients + directions (keep lowest id).
  Computed locally from RecipeNLG.csv; deletes stream to Supabase by id (no DB self-join).
Phase 2 — hybrid semantic duplicates:
  MiniLM embeddings → FAISS k-NN → hybrid score (semantic + ingredient Jaccard + title)
  → connected components → keep richest recipe per cluster (ingredient_count, then directions length).

Embeddings: uses local memmap export when available (fast); fetches any missing ids
from Supabase; falls back to Supabase-only when no local export exists.

By default this is a dry run. Pass --execute to DELETE rows from recipe.recipe_nlg
(cascades to recipe_nlg_features / recipe_nlg_embedding).

Usage:
  uv run python scripts/dedupe_recipe_nlg.py --dry-run
  uv run python scripts/dedupe_recipe_nlg.py --execute
  uv run python scripts/dedupe_recipe_nlg.py --limit 10000 --dry-run
  uv run python scripts/dedupe_recipe_nlg.py --use-db-embeddings --execute  # skip local
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from progress_utils import iter_progress
from recipe_dedup import (
    DedupConfig,
    exact_duplicate_ids,
    plan_hybrid_dedup,
    prepare_recipe_clean_frame,
)
from sample_recipes import DEFAULT_RECIPE_CSV

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "dedup"
DEFAULT_EMBED_DIR = ROOT / "Data" / "recipes" / "embeddings"
EMBED_FETCH_BATCH = 25_000


def local_embeddings_available(embed_dir: Path) -> bool:
    manifest_path = embed_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        emb_path = embed_dir / manifest.get("embeddings_path", "embeddings.f32.memmap")
        ids_path = embed_dir / manifest.get("recipe_ids_path", "recipe_ids.npy")
        return emb_path.is_file() and ids_path.is_file()
    except (json.JSONDecodeError, OSError):
        return False


def configure_session(cur) -> None:
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET lock_timeout = 0")


def fetch_supabase_ids(conn) -> set[int]:
    """Lightweight id list — recipe bodies come from local CSV."""
    with conn.cursor() as cur:
        configure_session(cur)
        cur.execute("SELECT id FROM recipe.recipe_nlg ORDER BY id")
        return {int(row[0]) for row in cur.fetchall()}


def load_local_recipes(
    id_set: set[int],
    *,
    csv_path: Path,
    limit: int | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Load title/ingredients/directions from local RecipeNLG.csv for Supabase ids."""
    ids = sorted(id_set)
    if limit is not None:
        ids = ids[:limit]
    id_filter = set(ids)
    tqdm.write(f"Scanning local CSV for {len(id_filter):,} recipe ids …")

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    for chunk in iter_progress(reader, desc="read RecipeNLG.csv", unit="chunk"):
        id_col = chunk.columns[0]
        sel = chunk[chunk[id_col].astype(int).isin(id_filter)]
        if len(sel):
            parts.append(sel)

    if not parts:
        raise RuntimeError(f"No matching rows in {csv_path}")

    recipes = pd.concat(parts, ignore_index=True)
    id_col = recipes.columns[0]
    out = recipes.rename(columns={id_col: "id"})
    return out.loc[:, ["id", "title", "ingredients", "directions"]].reset_index(drop=True)


def load_local_embedding_artifacts(embed_dir: Path) -> tuple[np.ndarray, np.memmap]:
    manifest = json.loads((embed_dir / "manifest.json").read_text())
    dims = int(manifest["dims"])
    count = int(manifest["count"])
    recipe_ids = np.load(embed_dir / "recipe_ids.npy")
    embeddings = np.memmap(
        embed_dir / manifest["embeddings_path"],
        dtype=np.float32,
        mode="r",
        shape=(count, dims),
    )
    if len(recipe_ids) != count:
        raise ValueError(f"recipe_ids length {len(recipe_ids)} != manifest count {count}")
    return recipe_ids, embeddings


def align_embeddings(
    recipe_ids_memmap: np.ndarray,
    embeddings: np.memmap,
    ordered_ids: np.ndarray,
) -> np.ndarray:
    """Reorder memmap rows to match clean_df id order."""
    pos = pd.Series(np.arange(len(recipe_ids_memmap)), index=recipe_ids_memmap.astype(np.int64))
    indices = pos.loc[ordered_ids.astype(np.int64)].to_numpy()
    return np.asarray(embeddings[indices], dtype=np.float32)


def build_hybrid_vectors(
    ordered_ids: np.ndarray,
    *,
    memmap_ids: np.ndarray | None,
    memmap_emb: np.memmap | None,
    db_vectors: dict[int, np.ndarray],
) -> np.ndarray:
    """Assemble embedding matrix: local memmap rows first, then Supabase fallbacks."""
    n = len(ordered_ids)
    if memmap_emb is not None:
        dims = memmap_emb.shape[1]
    elif db_vectors:
        dims = next(iter(db_vectors.values())).shape[0]
    else:
        raise RuntimeError("No embedding source available")

    out = np.empty((n, dims), dtype=np.float32)
    ids = ordered_ids.astype(np.int64)

    if memmap_ids is not None and memmap_emb is not None:
        pos = pd.Series(np.arange(len(memmap_ids)), index=memmap_ids.astype(np.int64))
        has_local = np.isin(ids, memmap_ids)
        if has_local.any():
            local_idx = pos.loc[ids[has_local]].to_numpy()
            out[has_local] = np.asarray(memmap_emb[local_idx], dtype=np.float32)
        need_db = ~has_local
    else:
        need_db = np.ones(n, dtype=bool)

    if need_db.any():
        db_positions = np.flatnonzero(need_db)
        for pos_i in db_positions:
            rid = int(ids[pos_i])
            vec = db_vectors.get(rid)
            if vec is None:
                raise RuntimeError(f"Missing embedding for recipe_id {rid}")
            out[pos_i] = vec

    return out


def _register_pgvector(conn) -> None:
    from pgvector.psycopg2 import register_vector

    register_vector(conn)


def fetch_embeddings_from_db(
    conn,
    recipe_ids: set[int] | None = None,
    *,
    batch_size: int = EMBED_FETCH_BATCH,
) -> dict[int, np.ndarray]:
    """Stream embeddings from Supabase; optional filter to a recipe id set."""
    _register_pgvector(conn)
    vectors: dict[int, np.ndarray] = {}

    if recipe_ids is not None and len(recipe_ids) == 0:
        return vectors

    if recipe_ids is None:
        last_id = -1
        while True:
            with conn.cursor() as cur:
                configure_session(cur)
                cur.execute(
                    """
                    SELECT e.recipe_id, e.embedding
                    FROM recipe.recipe_nlg_embedding e
                    INNER JOIN recipe.recipe_nlg r ON r.id = e.recipe_id
                    WHERE e.recipe_id > %s
                    ORDER BY e.recipe_id
                    LIMIT %s
                    """,
                    (last_id, batch_size),
                )
                rows = cur.fetchall()
            if not rows:
                break
            for rid, emb in rows:
                vectors[int(rid)] = np.asarray(emb, dtype=np.float32)
            last_id = int(rows[-1][0])
            if len(rows) < batch_size:
                break
            if len(vectors) % (batch_size * 4) == 0:
                tqdm.write(f"  fetched {len(vectors):,} embeddings from Supabase …", flush=True)
        return vectors

    id_list = sorted(recipe_ids)
    for start in iter_progress(
        range(0, len(id_list), batch_size),
        total=(len(id_list) + batch_size - 1) // batch_size,
        desc="fetch Supabase embeddings",
        unit="batch",
    ):
        chunk = id_list[start : start + batch_size]
        with conn.cursor() as cur:
            configure_session(cur)
            cur.execute(
                """
                SELECT e.recipe_id, e.embedding
                FROM recipe.recipe_nlg_embedding e
                INNER JOIN recipe.recipe_nlg r ON r.id = e.recipe_id
                WHERE e.recipe_id = ANY(%s)
                ORDER BY e.recipe_id
                """,
                (chunk,),
            )
            rows = cur.fetchall()
        for rid, emb in rows:
            vectors[int(rid)] = np.asarray(emb, dtype=np.float32)

    return vectors


def delete_ids_batched(
    conn,
    ids: list[int],
    batch_size: int,
    *,
    desc: str = "delete dupes",
) -> int:
    if not ids:
        return 0
    # psycopg2 cannot adapt numpy scalar types (e.g. int64 from pandas/numpy ids).
    py_ids = [int(x) for x in ids]
    deleted = 0
    sql = "DELETE FROM recipe.recipe_nlg WHERE id = ANY(%s)"
    batches = range(0, len(py_ids), batch_size)
    for start in tqdm(batches, desc=desc, unit="batch"):
        chunk = py_ids[start : start + batch_size]
        with conn.cursor() as cur:
            configure_session(cur)
            cur.execute(sql, (chunk,))
        conn.commit()
        deleted += len(chunk)
        tqdm.write(f"  deleted {deleted:,}/{len(py_ids):,} ids", flush=True)
    return deleted


def plan_exact_duplicates_local(
    supabase_ids: set[int],
    *,
    csv_path: Path,
) -> list[int]:
    """Find exact dupes locally (avoids heavy self-join on Supabase)."""
    print(f"Loading {len(supabase_ids):,} recipes from CSV for exact dedup …", flush=True)
    raw = load_local_recipes(supabase_ids, csv_path=csv_path)
    clean = prepare_recipe_clean_frame(raw)
    exact_ids = exact_duplicate_ids(clean)
    # Only delete rows that still exist in Supabase.
    return sorted(set(exact_ids) & supabase_ids)


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
        help="Force Supabase embeddings (skip local memmap even if present)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RECIPE_CSV,
        help=f"Local RecipeNLG CSV (default: {DEFAULT_RECIPE_CSV})",
    )
    parser.add_argument(
        "--embed-dir",
        type=Path,
        default=DEFAULT_EMBED_DIR,
        help=f"Local embedding export (default: {DEFAULT_EMBED_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N recipe ids (testing)",
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

    if not args.csv.is_file():
        sys.exit(f"Missing local CSV: {args.csv}")

    cfg = DedupConfig(
        k=args.k,
        semantic_pair_threshold=args.semantic_threshold,
        encode_batch_size=args.encode_batch_size,
    )

    t0 = time.perf_counter()
    with connect() as conn:
        print("Fetching recipe ids from Supabase …", flush=True)
        supabase_ids = fetch_supabase_ids(conn)
        print(f"  Supabase ids: {len(supabase_ids):,}", flush=True)

        exact_ids: list[int] = []
        if not args.skip_exact:
            exact_ids = plan_exact_duplicates_local(
                supabase_ids, csv_path=args.csv
            )
        print(f"Exact duplicate ids (phase 1, local): {len(exact_ids):,}", flush=True)

        if args.execute and exact_ids:
            print("Deleting exact duplicates (batched by id) …", flush=True)
            delete_ids_batched(
                conn, exact_ids, args.delete_batch_size, desc="delete exact dupes"
            )
            supabase_ids = fetch_supabase_ids(conn)
            print(f"  Supabase ids after exact pass: {len(supabase_ids):,}", flush=True)

        hybrid_delete_ids: list[int] = []
        keeper_ids: list[int] = []
        validated_pairs = pd.DataFrame()
        hybrid_input_rows = 0
        embedding_source = "none"

        if not args.skip_hybrid:
            memmap_ids: np.ndarray | None = None
            memmap_emb: np.memmap | None = None
            db_vectors: dict[int, np.ndarray] = {}

            use_local = (
                not args.use_db_embeddings
                and local_embeddings_available(args.embed_dir)
            )

            if use_local:
                print(f"Loading local embeddings from {args.embed_dir} …", flush=True)
                memmap_ids, memmap_emb = load_local_embedding_artifacts(args.embed_dir)
                local_set = set(int(x) for x in memmap_ids.tolist())
                work_ids = sorted(local_set & supabase_ids)
                missing_ids = supabase_ids - local_set
                if missing_ids:
                    print(
                        f"  {len(missing_ids):,} Supabase ids not in local export; "
                        "fetching from Supabase …",
                        flush=True,
                    )
                    db_vectors = fetch_embeddings_from_db(conn, missing_ids)
                    work_ids = sorted(set(work_ids) | set(db_vectors.keys()))
                embedding_source = "mixed" if db_vectors else "local"
                print(
                    f"  embedding source: {embedding_source} "
                    f"({len(work_ids):,} recipes with vectors)",
                    flush=True,
                )
            else:
                if args.use_db_embeddings:
                    print("Loading embeddings from Supabase (--use-db-embeddings) …", flush=True)
                else:
                    print(
                        f"No local export at {args.embed_dir}; "
                        "loading embeddings from Supabase …",
                        flush=True,
                    )
                db_vectors = fetch_embeddings_from_db(conn, supabase_ids)
                work_ids = sorted(db_vectors.keys())
                embedding_source = "supabase"
                print(f"  fetched {len(work_ids):,} embeddings from Supabase", flush=True)

            if args.limit is not None:
                work_ids = work_ids[: args.limit]

            n_no_emb = len(supabase_ids) - len(set(work_ids) & supabase_ids)
            if n_no_emb:
                tqdm.write(
                    f"  {n_no_emb:,} Supabase recipes lack embeddings; "
                    "skipped for hybrid (unchanged in DB)"
                )
            if not work_ids:
                sys.exit(
                    "No embedded recipes found. Run export_recipe_embeddings.py or "
                    "load_recipe_embeddings.py, or check Supabase."
                )

            print(f"Loading {len(work_ids):,} recipe rows from local CSV …", flush=True)
            raw = load_local_recipes(set(work_ids), csv_path=args.csv)
            hybrid_input_rows = len(raw)
            print("Preparing cleaned features …", flush=True)
            work_clean = prepare_recipe_clean_frame(raw)

            ordered = work_clean["id"].to_numpy(dtype=np.int64)
            if embedding_source == "local":
                vectors = align_embeddings(memmap_ids, memmap_emb, ordered)
            else:
                vectors = build_hybrid_vectors(
                    ordered,
                    memmap_ids=memmap_ids,
                    memmap_emb=memmap_emb,
                    db_vectors=db_vectors,
                )

            print("Planning hybrid deduplication …", flush=True)
            hybrid_delete_set, _hybrid_pairs, validated_pairs = plan_hybrid_dedup(
                work_clean, vectors, cfg
            )
            hybrid_delete_ids = sorted(hybrid_delete_set)
            keeper_ids = sorted(
                set(work_clean["id"].astype(int)) - hybrid_delete_set
            )
            print(f"Hybrid duplicate ids (phase 2): {len(hybrid_delete_ids):,}", flush=True)
            print(f"Keepers in embedded subset: {len(keeper_ids):,}", flush=True)

            if args.execute and hybrid_delete_ids:
                print("Deleting hybrid duplicates (streaming to Supabase) …", flush=True)
                delete_ids_batched(
                    conn,
                    hybrid_delete_ids,
                    args.delete_batch_size,
                    desc="delete hybrid dupes",
                )
        else:
            keeper_ids = sorted(supabase_ids)

        final_keepers = keeper_ids
        summary = {
            "mode": "execute" if args.execute else "dry-run",
            "supabase_rows": len(supabase_ids),
            "hybrid_input_rows": hybrid_input_rows,
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
                "embedding_source": embedding_source,
                "local_csv": str(args.csv),
                "embed_dir": str(args.embed_dir),
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
    print(f"\nArtifacts: {args.output_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"recipe.recipe_nlg rows now: {remaining:,}", flush=True)
    print(f"Done in {elapsed:.1f}s", flush=True)
    if args.dry_run:
        print("\nDry run only. Re-run with --execute to delete duplicates.", flush=True)


if __name__ == "__main__":
    main()
