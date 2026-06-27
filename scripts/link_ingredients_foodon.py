#!/usr/bin/env python3
"""Batch link USDA foods to FoodOn (tiered fuzzy + semantic + optional LLM).

Usage:
  uv run python scripts/build_foodon_index_cache.py
  uv run python scripts/build_foodon_embed_index.py
  uv run python scripts/link_ingredients_foodon.py --limit 1000
  uv run python scripts/link_ingredients_foodon.py --limit 500 --llm-model qwen/qwen3.6-35b-a3b --llm-url http://10.0.0.2:1234/v1
  uv run python scripts/link_ingredients_foodon.py --llm-model qwen/qwen3.6-35b-a3b --llm-url http://10.0.0.2:1234/v1 --checkpoint-every 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from diet_tags_io import load_foods_catalog
from foodon_embed_index import FoodOnEmbedIndex
from foodon_index import FoodOnIndex
from foodon_link_core import link_food_to_foodon
from foodon_link_judge import FoodOnLLMJudge
from foodon_mapping_io import LINKER_VERSION, load_mapping, mapping_path, merge_mapping, write_mapping
from foodon_paths import FOODON_EMBED_DIR, FOODON_INDEX_CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description="Link USDA fdc_id to FoodOn classes")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="Re-link fdc_ids already in mapping")
    parser.add_argument("--no-semantic", action="store_true", help="Fuzzy-only (no embedding index)")
    parser.add_argument("--llm-model", type=str, default=None, help="Local LLM model id for tier-3 judge")
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:1234/v1")
    parser.add_argument(
        "--llm-api",
        choices=("openai", "ollama"),
        default="openai",
        help="openai = LM Studio/vLLM /v1/chat/completions; ollama = /api/chat",
    )
    parser.add_argument("--ollama-model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ollama-url", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--review-only", action="store_true", help="Only export needs_review rows summary")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Write mapping every N newly linked foods (0 = only at end)",
    )
    args = parser.parse_args()

    llm_model = args.llm_model or args.ollama_model
    llm_url = args.llm_url
    if args.ollama_url:
        llm_url = args.ollama_url
        args.llm_api = "ollama"

    if not FOODON_INDEX_CACHE.is_file():
        print("Run: uv run python scripts/build_foodon_index_cache.py", flush=True)
        raise SystemExit(1)

    foodon = FoodOnIndex.from_cache(FOODON_INDEX_CACHE)
    embed_index = None
    if not args.no_semantic:
        if not (FOODON_EMBED_DIR / "manifest.json").is_file():
            print("Run: uv run python scripts/build_foodon_embed_index.py", flush=True)
            raise SystemExit(1)
        embed_index = FoodOnEmbedIndex.from_disk()

    llm_judge = None
    if llm_model:
        llm_judge = FoodOnLLMJudge(model=llm_model, base_url=llm_url, api=args.llm_api)
        print(f"LLM judge: {llm_model} @ {llm_url} ({args.llm_api})", flush=True)

    foods = load_foods_catalog(limit=args.limit)
    print(f"Catalog foods to scan: {len(foods):,}", flush=True)
    existing = load_mapping()
    existing_ids = set(existing["fdc_id"].astype(int).tolist()) if not existing.empty else set()
    if existing_ids and not args.refresh:
        print(f"Existing mapping rows (will skip): {len(existing_ids):,}", flush=True)

    rows: list[dict] = []
    pending: list[dict] = []
    skipped = 0
    t0 = time.perf_counter()

    def _flush_checkpoint(batch: list[dict], *, final: bool = False) -> None:
        if not batch:
            return
        current = load_mapping()
        merged = merge_mapping(current, pd.DataFrame(batch))
        out = write_mapping(merged)
        label = "Final" if final else "Checkpoint"
        print(
            f"  {label}: wrote {len(batch):,} rows -> {out} (total {len(merged):,}, {time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    for i, row in enumerate(foods.itertuples(index=False), start=1):
        fdc_id = int(row.fdc_id)
        if fdc_id in existing_ids and not args.refresh:
            skipped += 1
            continue
        desc = str(row.description)
        result = link_food_to_foodon(
            fdc_id,
            desc,
            foodon_index=foodon,
            embed_index=embed_index,
            llm_judge=llm_judge,
        )
        row_dict = result.to_row()
        rows.append(row_dict)
        pending.append(row_dict)
        existing_ids.add(fdc_id)

        if len(rows) % 100 == 0:
            print(
                f"  progress {i:,}/{len(foods):,} catalog | {len(rows):,} new | {skipped:,} skipped",
                flush=True,
            )
        if args.checkpoint_every and len(pending) >= args.checkpoint_every:
            _flush_checkpoint(pending)
            pending = []

    if not rows:
        print("No new rows to write (use --refresh to re-link).", flush=True)
        return

    new_df = pd.DataFrame(rows)
    if args.checkpoint_every:
        if pending:
            _flush_checkpoint(pending, final=True)
        merged = load_mapping()
        out = mapping_path()
    else:
        merged = merge_mapping(existing, new_df)
        out = write_mapping(merged)
    elapsed = time.perf_counter() - t0

    summary = {
        "linker_version": LINKER_VERSION,
        "new_rows": len(new_df),
        "total_rows": len(merged),
        "method_counts": new_df["match_method"].value_counts().to_dict(),
        "reviewed_count": int(new_df["reviewed"].sum()),
        "linked_count": int(new_df["foodon_id"].notna().sum()),
        "elapsed_s": round(elapsed, 1),
        "output": str(out),
    }
    summary_path = out.parent / "fdc_foodon_mapping_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(new_df):,} new links -> {out} ({elapsed:.1f}s)", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if args.review_only:
        review = merged[merged["reviewed"] == True]  # noqa: E712
        print(f"Needs review: {len(review):,}", flush=True)


if __name__ == "__main__":
    main()
