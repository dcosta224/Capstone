#!/usr/bin/env python3
"""Export recipe embeddings from Supabase pgvector to local memmap artifacts.

Reads recipe.recipe_nlg_embedding (384-d MiniLM, L2-normalized) in batches and
writes:
  Data/recipes/embeddings/recipe_ids.npy
  Data/recipes/embeddings/embeddings.f32.memmap  (N × dims, float32)
  Data/recipes/embeddings/manifest.json

Optimized for throughput:
  - One persistent DB connection (reconnect only on transient errors)
  - Native pgvector deserialization (no embedding::text)
  - Bulk numpy writes per batch (no per-row string parsing)

Idempotent: skips re-export when manifest row count matches the database unless
--force is passed. Checkpoints to .export_checkpoint.json for resume.

Usage:
  uv run python scripts/export_recipe_embeddings.py
  uv run python scripts/export_recipe_embeddings.py --force
  uv run python scripts/export_recipe_embeddings.py --batch-size 50000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "Data" / "recipes" / "embeddings"
BATCH_SIZE = 25_000


def configure_connection(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SET lock_timeout = 0")
    from pgvector.psycopg2 import register_vector

    register_vector(conn)


def batch_from_rows(rows: list[tuple], dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a fetchall page to int64 ids and (n, dims) float32 matrix."""
    n = len(rows)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, dims), dtype=np.float32)

    ids = np.fromiter((int(r[0]) for r in rows), dtype=np.int64, count=n)
    vectors = [r[1] for r in rows]
    first = vectors[0]
    if isinstance(first, np.ndarray):
        emb = np.stack(vectors).astype(np.float32, copy=False)
    else:
        emb = np.asarray(vectors, dtype=np.float32)

    if emb.shape != (n, dims):
        raise RuntimeError(f"Batch shape {emb.shape} != ({n}, {dims})")
    return ids, emb


def db_embedding_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM recipe.recipe_nlg_embedding")
        return int(cur.fetchone()[0])


def db_embedding_meta(conn) -> tuple[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model, dims
            FROM recipe.recipe_nlg_embedding
            GROUP BY model, dims
            ORDER BY count(*) DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("recipe.recipe_nlg_embedding is empty")
        return str(row[0]), int(row[1])


def load_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def save_checkpoint(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


class ExportConnection:
    """Persistent Supabase connection; reconnect only after OperationalError."""

    def __init__(self) -> None:
        self._conn = None

    def get(self):
        if self._conn is None or self._conn.closed:
            self._conn = connect()
            configure_connection(self._conn)
        return self._conn

    def reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def close(self) -> None:
        self.reset()


_FETCH_SQL = """
    SELECT recipe_id, embedding
    FROM recipe.recipe_nlg_embedding
    WHERE recipe_id > %s
    ORDER BY recipe_id
    LIMIT %s
"""


def fetch_batch(
    db: ExportConnection,
    last_id: int,
    batch_size: int,
    *,
    max_retries: int = 5,
) -> list[tuple]:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            conn = db.get()
            with conn.cursor() as cur:
                cur.execute(_FETCH_SQL, (last_id, batch_size))
                return cur.fetchall()
        except psycopg2.OperationalError as exc:
            db.reset()
            if attempt + 1 >= max_retries:
                raise
            print(f"  DB reconnect ({attempt + 1}/{max_retries}): {exc}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return []


def export_embeddings(
    out_dir: Path,
    *,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> dict:
    load_dotenv()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    ids_path = out_dir / "recipe_ids.npy"
    emb_path = out_dir / "embeddings.f32.memmap"
    checkpoint_path = out_dir / ".export_checkpoint.json"

    db = ExportConnection()
    try:
        conn = db.get()
        n_db = db_embedding_count(conn)
        if n_db == 0:
            raise RuntimeError(
                "No rows in recipe.recipe_nlg_embedding. "
                "Finish the embedding load on the load branch first."
            )
        model, dims = db_embedding_meta(conn)
    except Exception:
        db.close()
        raise

    existing = load_manifest(manifest_path)
    if (
        not force
        and existing
        and existing.get("count") == n_db
        and existing.get("dims") == dims
        and ids_path.is_file()
        and emb_path.is_file()
    ):
        print(f"Up to date: {n_db:,} embeddings in {out_dir} (use --force to re-export)")
        checkpoint_path.unlink(missing_ok=True)
        db.close()
        return existing

    checkpoint = None if force else load_checkpoint(checkpoint_path)
    if (
        checkpoint
        and checkpoint.get("n_db") == n_db
        and checkpoint.get("dims") == dims
        and checkpoint.get("model") == model
        and 0 < int(checkpoint.get("offset", 0)) < n_db
    ):
        offset = int(checkpoint["offset"])
        last_id = int(checkpoint["last_id"])
        print(f"Resuming export at {offset:,}/{n_db:,} (last_id={last_id})", flush=True)
    else:
        offset = 0
        last_id = -1
        if checkpoint_path.is_file() and not force:
            checkpoint_path.unlink()

    print(
        f"Exporting {n_db:,} embeddings (dims={dims}, model={model}, "
        f"batch={batch_size:,}) → {out_dir}",
        flush=True,
    )
    t0 = time.perf_counter()

    recipe_ids = np.empty(n_db, dtype=np.int64)
    emb_mode = "r+" if emb_path.is_file() and offset > 0 else "w+"
    embeddings = np.memmap(emb_path, dtype=np.float32, mode=emb_mode, shape=(n_db, dims))

    try:
        while offset < n_db:
            rows = fetch_batch(db, last_id, batch_size)
            if not rows:
                break

            ids, emb = batch_from_rows(rows, dims)
            n = len(rows)
            remaining = n_db - offset
            if n > remaining:
                ids = ids[:remaining]
                emb = emb[:remaining]
                n = remaining
            if n == 0:
                break
            recipe_ids[offset : offset + n] = ids
            embeddings[offset : offset + n] = emb

            offset += n
            last_id = int(rows[-1][0])
            embeddings.flush()
            save_checkpoint(
                checkpoint_path,
                {
                    "offset": offset,
                    "last_id": last_id,
                    "n_db": n_db,
                    "dims": dims,
                    "model": model,
                },
            )
            elapsed = time.perf_counter() - t0
            rate = offset / elapsed if elapsed > 0 else 0
            print(
                f"  {offset:,}/{n_db:,} ({100 * offset / n_db:.1f}%) | {rate:.0f} rows/s",
                flush=True,
            )
    finally:
        db.close()

    if offset != n_db:
        raise RuntimeError(f"Expected {n_db} rows, exported {offset}")

    embeddings.flush()
    np.save(ids_path, recipe_ids)
    checkpoint_path.unlink(missing_ok=True)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dims": dims,
        "count": n_db,
        "recipe_ids_path": ids_path.name,
        "embeddings_path": emb_path.name,
        "dtype": "float32",
        "batch_size": batch_size,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done in {manifest['elapsed_sec']}s → {out_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export recipe embeddings from Supabase.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Rows per keyset page (default {BATCH_SIZE:,})",
    )
    parser.add_argument("--force", action="store_true", help="Re-export even if manifest matches DB count.")
    args = parser.parse_args()
    export_embeddings(args.out_dir, batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()
