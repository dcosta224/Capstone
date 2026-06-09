#!/usr/bin/env python3
"""Compare baseline (v2) vs portion-aware (v3) LLM ingredient match runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from portion_gram import (
    build_count_portion_index,
    build_portion_index,
    resolve_grams_from_parsed_row,
)
from db import connect, load_dotenv
from parse_recipe_ingredient import parse_ingredient_fields

ROOT = Path(__file__).resolve().parents[1]


def _load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"recipe_id", "ingredient_idx", "ingredient", "llm_fdc_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _baseline_grams_for_row(row: pd.Series, volume_index: dict, count_index: dict) -> tuple:
    parsed = parse_ingredient_fields(str(row["ingredient"]))
    parsed["ingredient"] = row["ingredient"]
    result = resolve_grams_from_parsed_row(
        parsed,
        int(row["llm_fdc_id"]) if pd.notna(row.get("llm_fdc_id")) else None,
        portion_index=volume_index,
        count_portion_index=count_index,
    )
    return result.grams, result.status


def compare_runs(
    baseline: pd.DataFrame,
    portion: pd.DataFrame,
    *,
    volume_index: dict | None = None,
    count_index: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    keys = ["recipe_id", "ingredient_idx"]
    b = baseline.copy()
    p = portion.copy()
    b["llm_fdc_id"] = pd.to_numeric(b["llm_fdc_id"], errors="coerce")
    p["llm_fdc_id"] = pd.to_numeric(p["llm_fdc_id"], errors="coerce")

    merged = b.merge(
        p,
        on=keys,
        how="inner",
        suffixes=("_baseline", "_portion"),
    )
    merged["fdc_id_changed"] = merged["llm_fdc_id_baseline"] != merged["llm_fdc_id_portion"]
    if "llm_abstained_baseline" in merged.columns:
        merged["baseline_abstained"] = merged["llm_abstained_baseline"].astype(bool)
    else:
        merged["baseline_abstained"] = merged["llm_fdc_id_baseline"].isna()
    if "llm_abstained_portion" in merged.columns:
        merged["portion_abstained"] = merged["llm_abstained_portion"].astype(bool)
    else:
        merged["portion_abstained"] = merged["llm_fdc_id_portion"].isna()

    if volume_index is not None and count_index is not None:
        base_grams = []
        for _, row in merged.iterrows():
            g, s = _baseline_grams_for_row(
                pd.Series(
                    {
                        "ingredient": row.get("ingredient_baseline") or row.get("ingredient_portion"),
                        "llm_fdc_id": row["llm_fdc_id_baseline"],
                    }
                ),
                volume_index,
                count_index,
            )
            base_grams.append((g, s))
        merged["grams_baseline_recomputed"] = [g for g, _ in base_grams]
        merged["grams_status_baseline_recomputed"] = [s for _, s in base_grams]

    grams_portion_col = "grams_portion" if "grams_portion" in merged.columns else "grams"
    if grams_portion_col in merged.columns and "grams_baseline_recomputed" in merged.columns:
        merged["baseline_resolved"] = merged["grams_baseline_recomputed"].notna()
        merged["portion_resolved"] = merged[grams_portion_col].notna()
        merged["newly_resolved"] = (~merged["baseline_resolved"]) & merged["portion_resolved"]
        merged["lost_resolution"] = merged["baseline_resolved"] & (~merged["portion_resolved"])

    summary = {
        "n_joined": int(len(merged)),
        "fdc_id_agreement_rate": round(
            float((~merged["fdc_id_changed"]).mean()), 4
        )
        if len(merged)
        else None,
        "n_fdc_id_changed": int(merged["fdc_id_changed"].sum()),
        "baseline_abstain_rate": round(float(merged["baseline_abstained"].mean()), 4),
        "portion_abstain_rate": round(float(merged["portion_abstained"].mean()), 4),
    }

    if grams_portion_col in merged.columns and "baseline_resolved" in merged.columns:
        summary["baseline_gram_resolvable_rate"] = round(
            float(merged["baseline_resolved"].mean()), 4
        )
        summary["portion_gram_resolvable_rate"] = round(
            float(merged["portion_resolved"].mean()), 4
        )
        summary["n_newly_resolved"] = int(merged["newly_resolved"].sum())
        summary["n_lost_resolution"] = int(merged["lost_resolution"].sum())

    if "amount_kind_portion" in merged.columns:
        summary["amount_kind_counts"] = (
            merged["amount_kind_portion"].value_counts().to_dict()
        )
    if "retrieval_tier_portion" in merged.columns:
        summary["retrieval_tier_counts"] = (
            merged["retrieval_tier_portion"].value_counts().to_dict()
        )
    if "grams_status_portion" in merged.columns:
        summary["grams_status_portion_counts"] = (
            merged["grams_status_portion"].value_counts().to_dict()
        )

    matter = merged[
        merged.get("fdc_id_changed", False)
        | merged.get("newly_resolved", False)
        | merged.get("lost_resolution", False)
    ]
    summary["n_meaningful_differences"] = int(len(matter))

    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "scratch" / "recipe_matching_llm_100_baseline_rerun" / "ingredient_matches_llm.csv",
    )
    parser.add_argument(
        "--portion",
        type=Path,
        default=ROOT / "scratch" / "recipe_matching_llm_100_portion" / "ingredient_matches_llm.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "scratch" / "recipe_matching_llm_100_portion",
    )
    args = parser.parse_args()
    load_dotenv()

    baseline = _load_matches(args.baseline)
    portion = _load_matches(args.portion)

    with connect() as conn:
        volume_index = build_portion_index(conn)
        count_index = build_count_portion_index(conn)

    merged, summary = compare_runs(
        baseline, portion, volume_index=volume_index, count_index=count_index
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "comparison_merged.csv"
    summary_path = args.output_dir / "comparison_summary.json"
    merged.to_csv(merged_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {merged_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
