#!/usr/bin/env python3
"""Build semantic embedding index over FoodOn class labels.

Usage:
  uv run python scripts/build_foodon_embed_index.py
  uv run python scripts/build_foodon_embed_index.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_embed_index import FoodOnEmbedIndex
from foodon_index import FoodOnIndex
from foodon_paths import FOODON_EMBED_DIR, FOODON_INDEX_CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FoodOn label embedding index")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = FOODON_EMBED_DIR / "manifest.json"
    if manifest.is_file() and not args.force:
        idx = FoodOnEmbedIndex.from_disk()
        print(f"Index exists: {len(idx.node_ids):,} classes at {FOODON_EMBED_DIR}", flush=True)
        return

    if not FOODON_INDEX_CACHE.is_file():
        print("Run: uv run python scripts/build_foodon_index_cache.py", flush=True)
        raise SystemExit(1)

    t0 = time.perf_counter()
    foodon = FoodOnIndex.from_cache(FOODON_INDEX_CACHE)
    embed_idx = FoodOnEmbedIndex.build(foodon)
    out = embed_idx.save()
    elapsed = time.perf_counter() - t0
    print(f"Wrote {len(embed_idx.node_ids):,} embeddings to {out} in {elapsed:.1f}s", flush=True)

    sample = embed_idx.search("cheddar cheese", k=3)
    print("Sample search 'cheddar cheese':", flush=True)
    for hit in sample:
        print(f"  {hit['label']} ({hit['id']}) score={hit['score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
