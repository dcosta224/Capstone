"""Local artifacts and error-analysis helpers for parse experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from parse_recipe_ingredient import looks_parseable_raw

ERROR_ANALYSIS_COLUMNS = [
    "ingredient_raw",
    "recipe_id",
    "ingredient_idx",
    "pipeline_version",
    "library_parse_status",
    "library_quantity",
    "library_unit",
    "library_name",
    "rules_parse_status",
    "rules_quantity",
    "rules_unit",
    "rules_confidence",
    "llm_called",
    "llm_quantity",
    "llm_unit",
    "llm_name",
    "llm_measurable",
    "llm_certainty",
    "llm_rationale",
    "llm_error",
    "final_method",
    "final_parse_status",
    "final_quantity",
    "final_unit",
    "final_name",
    "error_category",
]


def manifest_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_error_analysis(
    results_df: pd.DataFrame,
    *,
    other_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter rows worth manual review."""
    df = results_df.copy()
    df["error_category"] = None

    categories: list[str] = []

    other_lookup: dict[tuple[int, int], pd.Series] = {}
    if other_df is not None and not other_df.empty:
        for row in other_df.itertuples(index=False):
            other_lookup[(int(row.recipe_id), int(row.ingredient_idx))] = row

    for row in df.itertuples(index=False):
        cats: list[str] = []
        raw = str(row.ingredient_raw)

        if row.final_parse_status != "ok" and looks_parseable_raw(raw):
            cats.append("unparsed_with_leading_qty")
        if row.llm_called and row.llm_certainty is not None and float(row.llm_certainty) < 0.7:
            cats.append("low_llm_certainty")
        if row.llm_error is not None and str(row.llm_error) != "nan":
            cats.append("llm_error")
        if row.final_parse_status == "llm_invalid":
            cats.append("llm_invalid")
        if row.final_method == "llm":
            cats.append("llm_resolved")

        key = (int(row.recipe_id), int(row.ingredient_idx))
        if key in other_lookup:
            other = other_lookup[key]
            if _final_fields_differ(row, other):
                cats.append("disagreement_vs_other")

        if row.pipeline_version == "lib_rules_llm" and row.final_method == "rule":
            lib_needs = row.library_parse_status in (
                "error",
                "ambiguous",
                "no_amount",
                "empty",
                "no_quantity",
                "quantity_no_known_unit",
            )
            if lib_needs and row.rules_parse_status in ("ok", "parsed_size_as_count"):
                cats.append("rules_rescued")

        categories.append("|".join(cats) if cats else None)

    df["error_category"] = categories
    flagged = df[df["error_category"].notna()].copy()

    rename = {
        "library_parse_status": "library_parse_status",
        "library_quantity": "library_quantity",
        "library_unit": "library_unit",
        "library_name": "library_name",
        "rules_parse_status": "rules_parse_status",
        "rules_quantity": "rules_quantity",
        "rules_unit": "rules_unit",
        "final_parse_status": "final_parse_status",
        "final_quantity": "final_quantity",
        "final_unit": "final_unit",
        "final_name": "final_name",
    }
    cols = [c for c in ERROR_ANALYSIS_COLUMNS if c in flagged.columns]
    return flagged[cols] if not flagged.empty else pd.DataFrame(columns=ERROR_ANALYSIS_COLUMNS)


def _final_fields_differ(a: Any, b: Any) -> bool:
    for field in ("final_quantity", "final_unit", "final_name"):
        va = getattr(a, field, None)
        vb = getattr(b, field, None)
        if pd.isna(va) and pd.isna(vb):
            continue
        if str(va).strip().lower() != str(vb).strip().lower():
            return True
    return False


def build_comparison(
    lib_llm_df: pd.DataFrame,
    lib_rules_df: pd.DataFrame,
    *,
    lib_llm_report: dict[str, Any],
    lib_rules_report: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cross-pipeline comparison tables and summary."""
    join_cols = ["recipe_id", "ingredient_idx", "ingredient_raw", "ingredient_norm"]
    a = lib_llm_df.add_suffix("_v1")
    b = lib_rules_df.add_suffix("_v2")

    a = a.rename(columns={f"{c}_v1": c for c in join_cols})
    b = b.rename(columns={f"{c}_v2": c for c in join_cols})

    side = a.merge(
        b,
        on=join_cols,
        how="outer",
    )

    disagree_mask = (
        (side["final_quantity_v1"].astype(str) != side["final_quantity_v2"].astype(str))
        | (side["final_unit_v1"].astype(str) != side["final_unit_v2"].astype(str))
        | (side["final_name_v1"].astype(str) != side["final_name_v2"].astype(str))
    )
    disagreements = side[disagree_mask].copy()

    rules_rescued = lib_rules_df[
        (lib_rules_df["final_method"] == "rule")
        & (lib_rules_df["library_parse_status"].isin([
            "error", "ambiguous", "no_amount", "empty", "no_quantity", "quantity_no_known_unit",
        ]))
    ].copy()

    if not rules_rescued.empty and not lib_llm_df.empty:
        v1_lookup = lib_llm_df.set_index(["recipe_id", "ingredient_idx"])
        rescued_idx = []
        for row in rules_rescued.itertuples(index=False):
            key = (int(row.recipe_id), int(row.ingredient_idx))
            if key in v1_lookup.index:
                v1_row = v1_lookup.loc[key]
                if v1_row["final_method"] == "llm" or v1_row["library_parse_status"] in (
                    "error", "ambiguous", "no_amount", "no_quantity", "quantity_no_known_unit",
                ):
                    rescued_idx.append(key)
        if rescued_idx:
            rules_rescued = rules_rescued[
                rules_rescued.apply(
                    lambda r: (int(r["recipe_id"]), int(r["ingredient_idx"])) in rescued_idx,
                    axis=1,
                )
            ]

    llm_only = side[
        (
            (side["final_method_v1"] == "llm")
            | (side["final_method_v2"] == "llm")
        )
        & (
            (side["final_parse_status_v1"] == "ok")
            | (side["final_parse_status_v2"] == "ok")
        )
    ].copy()

    merged_measurable = lib_rules_df[["recipe_id", "ingredient_idx", "final_measurable"]].merge(
        lib_llm_df[["recipe_id", "ingredient_idx", "final_measurable"]],
        on=["recipe_id", "ingredient_idx"],
        suffixes=("_v2", "_v1"),
        how="left",
    )
    v2_measurable = merged_measurable["final_measurable_v2"] == True  # noqa: E712
    v1_measurable = merged_measurable["final_measurable_v1"] == True  # noqa: E712
    v2_strictly_better = int((v2_measurable & ~v1_measurable).sum())

    summary = {
        "rules_rescued_count": int(len(rules_rescued)),
        "llm_calls_delta": int(
            lib_llm_report.get("n_llm_calls", 0) - lib_rules_report.get("n_llm_calls", 0)
        ),
        "disagreement_rate": round(float(len(disagreements) / max(len(side), 1)), 4),
        "v2_strictly_better_count": v2_strictly_better,
        "coverage_ok_rate_v1": lib_llm_report.get("coverage_ok_rate"),
        "coverage_ok_rate_v2": lib_rules_report.get("coverage_ok_rate"),
        "coverage_ok_delta_v2_minus_v1": round(
            float(lib_rules_report.get("coverage_ok_rate", 0))
            - float(lib_llm_report.get("coverage_ok_rate", 0)),
            4,
        ),
        "llm_call_rate_v1": lib_llm_report.get("llm_call_rate"),
        "llm_call_rate_v2": lib_rules_report.get("llm_call_rate"),
        "cost_total_usd_v1": lib_llm_report.get("cost_total_usd"),
        "cost_total_usd_v2": lib_rules_report.get("cost_total_usd"),
    }

    return side, rules_rescued, disagreements, llm_only, summary


def write_pipeline_artifacts(
    work_dir: Path,
    results_df: pd.DataFrame,
    llm_calls_df: pd.DataFrame,
    report: dict[str, Any],
    error_df: pd.DataFrame,
) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "parse_results": work_dir / "parse_results.parquet",
        "llm_calls": work_dir / "llm_calls.parquet",
        "parse_summary": work_dir / "parse_summary.json",
        "error_analysis": work_dir / "error_analysis.parquet",
    }
    results_df.to_parquet(paths["parse_results"], index=False)
    if llm_calls_df.empty:
        pd.DataFrame().to_parquet(paths["llm_calls"], index=False)
    else:
        llm_calls_df.to_parquet(paths["llm_calls"], index=False)
    paths["parse_summary"].write_text(json.dumps(report, indent=2, default=str) + "\n")
    if error_df.empty:
        pd.DataFrame(columns=ERROR_ANALYSIS_COLUMNS).to_parquet(paths["error_analysis"], index=False)
    else:
        error_df.to_parquet(paths["error_analysis"], index=False)
    return paths


def write_comparison_artifacts(
    comp_dir: Path,
    side: pd.DataFrame,
    rules_rescued: pd.DataFrame,
    disagreements: pd.DataFrame,
    llm_only: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Path]:
    comp_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "side_by_side": comp_dir / "side_by_side.parquet",
        "rules_rescued": comp_dir / "rules_rescued.parquet",
        "disagreements": comp_dir / "disagreements.parquet",
        "llm_only_fixes": comp_dir / "llm_only_fixes.parquet",
        "comparison_summary": comp_dir / "comparison_summary.json",
    }
    side.to_parquet(paths["side_by_side"], index=False)
    rules_rescued.to_parquet(paths["rules_rescued"], index=False)
    disagreements.to_parquet(paths["disagreements"], index=False)
    llm_only.to_parquet(paths["llm_only_fixes"], index=False)
    paths["comparison_summary"].write_text(json.dumps(summary, indent=2) + "\n")
    return paths
