#!/usr/bin/env python3
"""Export recipe embeddings from Supabase pgvector to local memmap artifacts.

Reads recipe.recipe_nlg_embedding (384-d MiniLM, L2-normalized) in batches and
writes:
  Data/recipes/embeddings/recipe_ids.npy
  Data/recipes/embeddings/embeddings.f32.memmap  (N × dims, float32)
  Data/recipes/embeddings/manifest.json

Idempotent: skips re-export when manifest row count matches the database unless
--force is passed.

Usage:
  uv run python scripts/export_recipe_embeddings.py
  uv run python scripts/export_recipe_embeddings.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "Data" / "recipes" / "embeddings"
EMBEDDING_DIMS = 384
BATCH_SIZE = 10_000


def parse_vector_text(raw: object) -> np.ndarray:
    """Parse pgvector text or list into float32 vector."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=np.float32)
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return np.fromiter((float(p) for p in parts), dtype=np.float32, count=len(parts))


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

    with connect() as conn:
        n_db = db_embedding_count(conn)
        if n_db == 0:
            raise RuntimeError(
                "No rows in recipe.recipe_nlg_embedding. "
                "Finish the embedding load on the load branch first."
            )
        model, dims = db_embedding_meta(conn)

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
            return existing

        print(f"Exporting {n_db:,} embeddings (dims={dims}, model={model}) → {out_dir}", flush=True)
        t0 = time.perf_counter()

        recipe_ids = np.empty(n_db, dtype=np.int64)
        embeddings = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_db, dims))

        offset = 0
        last_id = -1
        while offset < n_db:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT recipe_id, embedding::text
                    FROM recipe.recipe_nlg_embedding
                    WHERE recipe_id > %s
                    ORDER BY recipe_id
                    LIMIT %s
                    """,
                    (last_id, batch_size),
                )
                rows = cur.fetchall()
            if not rows:
                break

            for i, (recipe_id, emb_text) in enumerate(rows):
                recipe_ids[offset + i] = int(recipe_id)
                embeddings[offset + i] = parse_vector_text(emb_text)

            offset += len(rows)
            last_id = int(rows[-1][0])
            elapsed = time.perf_counter() - t0
            rate = offset / elapsed if elapsed > 0 else 0
            print(f"  {offset:,}/{n_db:,} ({100 * offset / n_db:.1f}%) | {rate:.0f} rows/s", flush=True)

        if offset != n_db:
            raise RuntimeError(f"Expected {n_db} rows, exported {offset}")

        embeddings.flush()
        np.save(ids_path, recipe_ids)

        manifest = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "dims": dims,
            "count": n_db,
            "recipe_ids_path": ids_path.name,
            "embeddings_path": emb_path.name,
            "dtype": "float32",
            "elapsed_sec": round(time.perf_counter() - t0, 1),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Done in {manifest['elapsed_sec']}s → {out_dir}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export recipe embeddings from Supabase.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Re-export even if manifest matches DB count.")
    args = parser.parse_args()
    export_embeddings(args.out_dir, batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()
