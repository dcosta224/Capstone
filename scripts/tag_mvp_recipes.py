#!/usr/bin/env python3
"""Export MVP resolved lines and write recipe diet tags to scratch/tag/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from db import load_dotenv
from diet_tags_corpus import build_recipe_diet_tags_for_corpus
from diet_tags_io import write_table
from mvp_corpus_cache import load_corpus_from_disk, warm_mvp_corpus

OUT_DIR = ROOT / "scratch" / "tag"


def _corpus_to_resolved_lines(corpus: dict) -> pd.DataFrame:
    rows: list[dict] = []
    names = {
        int(rid): str(name)
        for rid, name in zip(corpus["recipe_ids"], corpus["recipe_names"], strict=True)
    }
    for recipe_id, ing_df in (corpus.get("ingredients_by_recipe") or {}).items():
        rid = int(recipe_id)
        recipe_name = names.get(rid, str(rid))
        for row in ing_df.itertuples(index=False):
            if row.fdc_id is None or pd.isna(row.fdc_id):
                continue
            rows.append(
                {
                    "recipe_id": rid,
                    "recipe_name": recipe_name,
                    "fdc_id": int(row.fdc_id),
                    "gram_weight": float(row.gram_weight)
                    if row.gram_weight is not None and pd.notna(row.gram_weight)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag MVP corpus recipes with diet_tags.json")
    parser.add_argument("--refresh", action="store_true", help="Rebuild corpus cache from DB")
    parser.add_argument("--no-foodon", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    corpus = warm_mvp_corpus(force_refresh=args.refresh) if args.refresh else None
    if corpus is None:
        corpus = load_corpus_from_disk()
    if corpus is None:
        corpus = warm_mvp_corpus()

    resolved = _corpus_to_resolved_lines(corpus)
    write_table(resolved, OUT_DIR / "mvp_resolved_lines.parquet")

    recipe_diet_tags, recipe_restrictions = build_recipe_diet_tags_for_corpus(
        corpus["ingredients_by_recipe"],
        corpus["food_nutrients"],
        use_foodon=not args.no_foodon,
    )

    wide_rows: list[dict] = []
    for rid in corpus["recipe_ids"]:
        rid = int(rid)
        row = {"recipe_id": rid}
        for tslug, val in (recipe_diet_tags.get(rid) or {}).items():
            if val is not None:
                row[f"tag_{tslug}"] = bool(val)
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    write_table(wide_df, OUT_DIR / "recipe_diet_tags_wide.parquet")

    summary = {
        "recipes": len(wide_df),
        "tag_true_counts": {
            col.replace("tag_", ""): int(wide_df[col].sum())
            for col in wide_df.columns
            if col.startswith("tag_") and wide_df[col].dtype == bool
        },
        "restriction_counts": {
            slug: sum(1 for restr in recipe_restrictions.values() if slug in restr)
            for slug in sorted({s for restr in recipe_restrictions.values() for s in restr})
        },
    }
    (OUT_DIR / "recipe_diet_tags_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Resolved lines: {len(resolved):,}", flush=True)
    print(f"Recipes tagged: {len(wide_df):,}", flush=True)
    print(f"Wrote {OUT_DIR / 'mvp_resolved_lines.parquet'}", flush=True)
    print(f"Wrote {OUT_DIR / 'recipe_diet_tags_wide.parquet'}", flush=True)


if __name__ == "__main__":
    main()
