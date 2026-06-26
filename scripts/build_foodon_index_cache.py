#!/usr/bin/env python3
"""Build local FoodOn search index from Data/foodon-master/foodon.owl.

Usage:
  uv run python scripts/build_foodon_index_cache.py
  uv run python scripts/build_foodon_index_cache.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_paths import FOODON_INDEX_CACHE, FOODON_WEB_CACHE, resolve_owl_path
from foodon_index import FoodOnIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FoodOn index JSON cache")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = parser.parse_args()

    owl_path = resolve_owl_path()
    print(f"OWL: {owl_path}", flush=True)
    t0 = time.perf_counter()
    index = FoodOnIndex.from_owl(
        owl_path=owl_path,
        cache_path=FOODON_INDEX_CACHE,
        force_rebuild=args.force,
    )
    elapsed = time.perf_counter() - t0
    print(f"Classes indexed: {len(index.labels):,}", flush=True)
    print(f"Roots: {len(index.roots):,}", flush=True)
    print(f"Cache: {FOODON_INDEX_CACHE}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)

    FOODON_WEB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FOODON_INDEX_CACHE, FOODON_WEB_CACHE)
    print(f"Copied to: {FOODON_WEB_CACHE}", flush=True)

    sample = index.search("cheddar cheese", limit=3)
    print("Sample search 'cheddar cheese':", flush=True)
    for hit in sample:
        print(f"  {hit['label']} ({hit['id']}) score={hit['score']}", flush=True)


if __name__ == "__main__":
    main()
