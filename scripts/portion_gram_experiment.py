#!/usr/bin/env python3
"""Batch experiment: parse + fdc match + portion-to-gram resolution.

Modes:
  existing_llm — reuse llm_fdc_id from a prior ingredient_match_llm CSV or Supabase
  inline_full  — sample recipes, staged match, then resolve grams

Usage:
  uv run python scripts/portion_gram_experiment.py --mode existing_llm --limit 500
  uv run python scripts/portion_gram_experiment.py --mode existing_llm \\
      --input path/to/ingredient_matches_llm.csv
  uv run python scripts/portion_gram_experiment.py --mode inline_full --n-recipes 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv
from ingredient_match_staged import StagedFoodIndex, match_query, query_from_parsed_row
from ingredient_query_cache import (
    DEFAULT_WORK_DIR as MATCH_CACHE_DIR,
    load_or_build_food_artifacts,
    load_or_build_recipe_artifacts,
)
from load_food_4macro import load_food_4macro
from portion_gram import (
    build_count_portion_index,
    build_portion_index,
    resolve_grams,
    resolve_quantity_fields,
)
from progress_utils import iter_progress
from sample_recipes import explode_recipe_ingredients, load_recipes_by_id, sample_recipe_ids

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Data" / "portion_gram_experiment"
DEFAULT_LLM_CSV = ROOT / "scratch" / "recipe_matching_llm_100" / "ingredient_matches_llm.csv"

LLM_CSV_GLOB_CANDIDATES = [
    DEFAULT_LLM_CSV,
    ROOT / "mlartifacts" / "4413d7fc582149d5b7ff3bf9a8cb9ce2" / "artifacts" / "ingredient_matches_llm.csv",
]


def _find_default_llm_csv() -> Path | None:
    for path in LLM_CSV_GLOB_CANDIDATES:
        if path.is_file():
            return path
    for path in sorted((ROOT / "mlartifacts").glob("*/artifacts/ingredient_matches_llm.csv")):
        if path.is_file():
            return path
    return None


def load_existing_llm_rows(
    *,
    input_path: Path | None,
    limit: int | None,
    conn,
) -> pd.DataFrame:
    if input_path is not None and input_path.is_file():
        df = pd.read_csv(input_path)
        source = str(input_path)
    else:
        default = _find_default_llm_csv()
        if default is not None:
            df = pd.read_csv(default)
            source = str(default)
        else:
            sql = """
                SELECT recipe_id, ingredient_idx, ingredient, unit, llm_fdc_id
                FROM inference.match_inferences_0
                ORDER BY ts DESC
            """
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            with conn.cursor() as cur:
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            if not rows:
                raise RuntimeError(
                    "No existing LLM input found. Pass --input CSV or run inline_full mode."
                )
            df = pd.DataFrame(rows, columns=columns)
            source = "inference.match_inferences_0"

    required = {"recipe_id", "ingredient_idx", "ingredient", "llm_fdc_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns {sorted(missing)} (from {source})")

    df = df.copy()
    df["fdc_id"] = pd.to_numeric(df["llm_fdc_id"], errors="coerce")
    df["input_source"] = source
    if limit is not None:
        df = df.head(int(limit))
    return df.reset_index(drop=True)


def run_existing_llm_mode(
    df: pd.DataFrame,
    portion_index: dict,
    count_portion_index: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for row in iter_progress(
        df.itertuples(index=False),
        total=len(df),
        desc="resolve grams",
        unit="row",
    ):
        parsed = resolve_quantity_fields(str(row.ingredient), method="rules")
        t0 = time.perf_counter()
        result = resolve_grams(
            int(row.fdc_id) if pd.notna(row.fdc_id) else None,
            parsed.get("quantity"),
            parsed.get("unit"),
            name=parsed.get("name"),
            ingredient_raw=str(row.ingredient),
            amount_kind=parsed.get("amount_kind"),
            portion_index=portion_index,
            count_portion_index=count_portion_index,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "mode": "existing_llm",
                "recipe_id": int(row.recipe_id),
                "ingredient_idx": int(row.ingredient_idx),
                "ingredient": str(row.ingredient),
                "fdc_id": int(row.fdc_id) if pd.notna(row.fdc_id) else None,
                "parsed_quantity": parsed.get("quantity"),
                "parsed_unit": parsed.get("unit"),
                "parse_status": parsed.get("parse_status"),
                "parsed_unit_kind": parsed.get("amount_kind") or parsed.get("unit_kind"),
                "grams": result.grams,
                "status": result.status,
                "unit_kind": result.unit_kind,
                "portion_id": result.portion_id,
                "portion_ref_amount": result.portion_ref_amount,
                "portion_ref_unit": result.portion_ref_unit,
                "method": result.method,
                "resolve_ms": round(elapsed_ms, 2),
                "input_source": getattr(row, "input_source", None),
            }
        )
    return pd.DataFrame(rows)


def _build_staged_index(work_dir: Path) -> StagedFoodIndex:
    food_df = load_food_4macro()
    food_parsed, food_name_emb, food_prep_emb, food_dequant_emb, _meta = (
        load_or_build_food_artifacts(food_df, work_dir=work_dir)
    )
    return StagedFoodIndex.from_catalog(
        food_parsed,
        name_embeddings=food_name_emb,
        prep_embeddings=food_prep_emb,
        dequant_embeddings=food_dequant_emb,
    )


def run_inline_full_mode(
    *,
    n_recipes: int,
    seed: int,
    portion_index: dict,
    count_portion_index: dict,
    work_dir: Path,
    limit: int | None,
) -> pd.DataFrame:
    recipe_ids = sample_recipe_ids(n=n_recipes, seed=seed)
    recipes = load_recipes_by_id(recipe_ids)
    if recipes.empty:
        raise RuntimeError("No recipes loaded for inline_full mode")

    ingredients = explode_recipe_ingredients(recipes)
    if limit is not None:
        ingredients = ingredients.head(int(limit))

    parsed_df, name_emb, prep_emb, dequant_emb, _meta = load_or_build_recipe_artifacts(
        ingredients,
        work_dir=work_dir,
    )
    merged = ingredients.merge(
        parsed_df,
        on=["recipe_id", "ingredient_idx", "ingredient"],
        how="left",
    ).reset_index(drop=True)
    index = _build_staged_index(work_dir)

    rows: list[dict] = []
    for i in iter_progress(
        range(len(merged)),
        total=len(merged),
        desc="inline match+resolve",
        unit="row",
    ):
        row = merged.iloc[i]
        qrow = query_from_parsed_row(
            row,
            name_emb[i],
            prep_emb[i],
            dequant_emb[i],
        )
        match = match_query(qrow, index)
        fdc_id = match.get("matched_fdc_id")

        rules = resolve_quantity_fields(str(row["ingredient"]), method="rules")
        quantity = rules.get("quantity") if rules.get("quantity") is not None else row.get("quantity")
        unit = rules.get("unit") if rules.get("unit") else row.get("unit")

        t0 = time.perf_counter()
        result = resolve_grams(
            int(fdc_id) if fdc_id is not None and pd.notna(fdc_id) else None,
            quantity,
            unit,
            name=rules.get("name") or row.get("name"),
            ingredient_raw=str(row["ingredient"]),
            amount_kind=rules.get("amount_kind"),
            portion_index=portion_index,
            count_portion_index=count_portion_index,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "mode": "inline_full",
                "recipe_id": int(row["recipe_id"]),
                "ingredient_idx": int(row["ingredient_idx"]),
                "ingredient": str(row["ingredient"]),
                "fdc_id": int(fdc_id) if fdc_id is not None and pd.notna(fdc_id) else None,
                "matched_description": match.get("matched_description"),
                "match_score": match.get("match_score"),
                "match_quality": match.get("match_quality"),
                "parsed_quantity": quantity,
                "parsed_unit": unit,
                "parse_status": rules.get("parse_status"),
                "parsed_unit_kind": rules.get("unit_kind"),
                "grams": result.grams,
                "status": result.status,
                "unit_kind": result.unit_kind,
                "portion_id": result.portion_id,
                "portion_ref_amount": result.portion_ref_amount,
                "portion_ref_unit": result.portion_ref_unit,
                "method": result.method,
                "resolve_ms": round(elapsed_ms, 2),
                "input_source": "inline_staged",
            }
        )
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> dict:
    total = len(results)
    resolved = int(results["grams"].notna().sum())
    by_status = results["status"].value_counts().sort_index().to_dict()
    by_status = {str(k): int(v) for k, v in by_status.items()}

    unit_kind_stats: dict[str, dict] = {}
    for kind, group in results.groupby("parsed_unit_kind", dropna=False):
        key = str(kind) if kind is not None and str(kind) != "nan" else "null"
        n = len(group)
        ok = int(group["grams"].notna().sum())
        unit_kind_stats[key] = {
            "n": n,
            "resolved": ok,
            "coverage": round(ok / n, 4) if n else 0.0,
        }

    volume_rows = results[results["parsed_unit_kind"] == "volume"]
    mass_rows = results[results["parsed_unit_kind"] == "mass"]
    count_rows = results[results["parsed_unit_kind"] == "count"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": total,
        "n_resolved": resolved,
        "coverage_overall": round(resolved / total, 4) if total else 0.0,
        "coverage_volume": round(
            int(volume_rows["grams"].notna().sum()) / len(volume_rows), 4
        )
        if len(volume_rows)
        else None,
        "coverage_mass": round(int(mass_rows["grams"].notna().sum()) / len(mass_rows), 4)
        if len(mass_rows)
        else None,
        "coverage_count": round(int(count_rows["grams"].notna().sum()) / len(count_rows), 4)
        if len(count_rows)
        else None,
        "by_status": by_status,
        "by_parsed_unit_kind": unit_kind_stats,
        "mean_resolve_ms": round(float(results["resolve_ms"].mean()), 3)
        if len(results)
        else 0.0,
    }


def write_audit_sample(results: pd.DataFrame, path: Path, n: int = 200) -> None:
    if results.empty:
        path.write_text("recipe_id,ingredient_idx,ingredient,fdc_id,parsed_quantity,parsed_unit,grams,status,method\n")
        return

    parts: list[pd.DataFrame] = []
    per_status = max(1, n // max(len(results["status"].unique()), 1))
    for status, group in results.groupby("status"):
        parts.append(group.head(per_status))
    sample = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["recipe_id", "ingredient_idx"]
    )
    if len(sample) < n:
        extra = results[~results.index.isin(sample.index)].head(n - len(sample))
        sample = pd.concat([sample, extra], ignore_index=True)
    sample = sample.head(n)

    cols = [
        "recipe_id",
        "ingredient_idx",
        "ingredient",
        "fdc_id",
        "parsed_quantity",
        "parsed_unit",
        "parsed_unit_kind",
        "grams",
        "status",
        "portion_id",
        "method",
    ]
    cols = [c for c in cols if c in sample.columns]
    sample.loc[:, cols].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("existing_llm", "inline_full"),
        default="existing_llm",
    )
    parser.add_argument("--input", type=Path, default=None, help="ingredient_matches_llm.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--n-recipes", type=int, default=50, help="inline_full only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=MATCH_CACHE_DIR,
        help="Embedding cache dir for inline_full",
    )
    args = parser.parse_args()
    load_dotenv()

    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with connect() as conn:
        print("Building portion indices from usda.food_portion …", flush=True)
        portion_index = build_portion_index(conn)
        count_portion_index = build_count_portion_index(conn)
        n_foods = len(portion_index)
        n_portions = sum(len(v) for v in portion_index.values())
        n_count_foods = len(count_portion_index)
        n_count_portions = sum(len(v) for v in count_portion_index.values())
        print(
            f"  {n_portions:,} volume portions across {n_foods:,} fdc_ids; "
            f"{n_count_portions:,} count portions across {n_count_foods:,} fdc_ids",
            flush=True,
        )

        if args.mode == "existing_llm":
            print("Loading existing LLM match rows …", flush=True)
            input_df = load_existing_llm_rows(
                input_path=args.input,
                limit=args.limit,
                conn=conn,
            )
            print(f"  {len(input_df):,} rows", flush=True)
            results = run_existing_llm_mode(input_df, portion_index, count_portion_index)
        else:
            print(
                f"inline_full: {args.n_recipes} recipes (limit {args.limit} ingredients) …",
                flush=True,
            )
            results = run_inline_full_mode(
                n_recipes=args.n_recipes,
                seed=args.seed,
                portion_index=portion_index,
                count_portion_index=count_portion_index,
                work_dir=args.work_dir,
                limit=args.limit,
            )

    summary = build_summary(results)
    summary["mode"] = args.mode
    summary["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    summary["portion_index_fdc_ids"] = n_foods
    summary["portion_index_rows"] = n_portions
    summary["count_portion_index_fdc_ids"] = n_count_foods
    summary["count_portion_index_rows"] = n_count_portions

    results_path = args.output_dir / "results.parquet"
    summary_path = args.output_dir / "summary.json"
    audit_path = args.output_dir / "audit_sample.csv"

    results.to_parquet(results_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_audit_sample(results, audit_path)

    print(f"\nWrote {results_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {audit_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
