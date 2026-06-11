#!/usr/bin/env python3
"""Pre-build MVP demo corpus cache (recipes, embeddings, ingredients)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import load_dotenv
from mvp_corpus_cache import corpus_status, warm_mvp_corpus

if __name__ == "__main__":
    load_dotenv()
    corpus = warm_mvp_corpus(force_refresh="--refresh" in sys.argv)
    n_mvp = corpus.get("n_mvp_ids", corpus["n_recipes"])
    print(f"Cached {corpus['n_recipes']}/{n_mvp} MVP recipes ({corpus.get('emb_source')})")
    missing = corpus.get("missing_embedding_ids") or []
    if missing:
        print(f"WARNING: missing embeddings for recipe ids: {missing}", file=sys.stderr)
    print(f"Saved to mvp_web/cache/ — build took {corpus.get('build_ms')} ms")
    print(corpus_status())
