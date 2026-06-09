"""Tiered ingredient parsing pipelines for parse experiments."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ingredient_parse_llm import (
    DEFAULT_MODEL,
    normalize_ingredient_key,
    run_llm_parses_sync,
)
from parse_recipe_ingredient import (
    PARSE_FIELDS,
    finalize_parse_result,
    has_usable_amount,
    should_force_unmeasurable,
    merge_parse_result,
    needs_llm_fallback,
    parse_ingredient_fields,
)
from recipe_parse_rules import rule_parse_fields, rules_accepted

RULES_CONFIDENCE_THRESHOLD = 0.80


def _prefix_fields(result: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in PARSE_FIELDS:
        out[f"{prefix}_{field}"] = result.get(field)
    if prefix == "library":
        out["library_confidence"] = result.get("confidence")
    if prefix == "rules":
        out["rules_confidence"] = result.get("confidence")
    if prefix == "llm":
        out["llm_confidence"] = result.get("confidence")
        out["llm_measurable"] = result.get("measurable")
    if prefix == "final":
        out["final_confidence"] = result.get("confidence")
        out["final_measurable"] = result.get("measurable")
    return out


def _line_base(*, recipe_id: int, ingredient_idx: int, ingredient: str) -> dict[str, Any]:
    return {
        "recipe_id": int(recipe_id),
        "ingredient_idx": int(ingredient_idx),
        "ingredient_raw": str(ingredient),
        "ingredient_norm": normalize_ingredient_key(str(ingredient)),
    }


def _apply_llm(
    ingredient_raw: str,
    current: dict[str, Any],
    llm_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    key = normalize_ingredient_key(ingredient_raw)
    llm_row = llm_cache.get(key)
    if llm_row is None:
        return current, None

    llm_fields = llm_row["fields"]
    merged = merge_parse_result(current, llm_fields, final_method="llm")
    final = finalize_parse_result(merged, ingredient_raw=ingredient_raw, final_method="llm")
    return final, llm_row


def run_lib_llm(
    eval_df: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
    use_llm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pipeline A: library → LLM fallback."""
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    llm_needed_raw: list[str] = []

    for row in eval_df.itertuples(index=False):
        ingredient_raw = str(row.ingredient)
        library = parse_ingredient_fields(ingredient_raw)
        library = finalize_parse_result(
            {**library, "confidence": None, "measurable": None, "final_method": None},
            ingredient_raw=ingredient_raw,
            final_method="library",
        )

        if use_llm and needs_llm_fallback(library, ingredient_raw=ingredient_raw):
            llm_needed_raw.append(ingredient_raw)

    llm_cache: dict[str, dict[str, Any]] = {}
    llm_call_rows: list[dict[str, Any]] = []
    if use_llm and llm_needed_raw:
        llm_cache, llm_call_rows = run_llm_parses_sync(
            llm_needed_raw,
            model=model,
            concurrency=concurrency,
        )

    for row in eval_df.itertuples(index=False):
        ingredient_raw = str(row.ingredient)
        base = _line_base(
            recipe_id=int(row.recipe_id),
            ingredient_idx=int(row.ingredient_idx),
            ingredient=ingredient_raw,
        )
        library = parse_ingredient_fields(ingredient_raw)
        library = finalize_parse_result(
            {**library, "confidence": None, "measurable": None, "final_method": None},
            ingredient_raw=ingredient_raw,
            final_method="library",
        )

        rules_empty = {f: None for f in PARSE_FIELDS}
        rules_empty["parse_status"] = None
        rules_empty["parse_method"] = None
        rules_result = {**rules_empty, "confidence": None}

        llm_row: dict[str, Any] | None = None
        final = library
        llm_called = False

        if should_force_unmeasurable(library, ingredient_raw=ingredient_raw):
            final = finalize_parse_result(
                {**library, "measurable": False, "parse_status": "unmeasurable"},
                ingredient_raw=ingredient_raw,
                final_method="library",
            )
        elif needs_llm_fallback(library, ingredient_raw=ingredient_raw):
            if use_llm:
                final, llm_row = _apply_llm(ingredient_raw, library, llm_cache)
                llm_called = llm_row is not None
            else:
                final = finalize_parse_result(library, ingredient_raw=ingredient_raw, final_method="library")
        else:
            final = finalize_parse_result(library, ingredient_raw=ingredient_raw, final_method="library")

        record = {
            **base,
            **_prefix_fields(library, "library"),
            **_prefix_fields(rules_result, "rules"),
            "llm_called": llm_called,
            "llm_certainty": llm_row.get("llm_certainty") if llm_row else None,
            "llm_rationale": llm_row.get("llm_rationale") if llm_row else None,
            "llm_error": llm_row.get("llm_error") if llm_row else None,
            **_prefix_fields(llm_row["fields"] if llm_row else {f: None for f in PARSE_FIELDS}, "llm"),
            **_prefix_fields(final, "final"),
            "final_method": final.get("final_method"),
            "pipeline_version": "lib_llm",
        }
        rows.append(record)

    results_df = pd.DataFrame(rows)
    llm_calls_df = pd.DataFrame(llm_call_rows) if llm_call_rows else pd.DataFrame()
    meta = {
        "pipeline_version": "lib_llm",
        "use_llm": use_llm,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_llm_calls": len(llm_calls_df),
    }
    return results_df, llm_calls_df, meta


def run_lib_rules_llm(
    eval_df: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
    use_llm: bool = True,
    rules_threshold: float = RULES_CONFIDENCE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pipeline B: library → rules → LLM fallback."""
    t0 = time.perf_counter()
    llm_needed_raw: list[str] = []

    for row in eval_df.itertuples(index=False):
        ingredient_raw = str(row.ingredient)
        library = parse_ingredient_fields(ingredient_raw)
        if not needs_llm_fallback(library, ingredient_raw=ingredient_raw):
            continue

        rules = rule_parse_fields(ingredient_raw)
        if rules_accepted(rules, threshold=rules_threshold):
            continue
        if use_llm:
            llm_needed_raw.append(ingredient_raw)

    llm_cache: dict[str, dict[str, Any]] = {}
    llm_call_rows: list[dict[str, Any]] = []
    if use_llm and llm_needed_raw:
        llm_cache, llm_call_rows = run_llm_parses_sync(
            llm_needed_raw,
            model=model,
            concurrency=concurrency,
        )

    rows: list[dict[str, Any]] = []
    for row in eval_df.itertuples(index=False):
        ingredient_raw = str(row.ingredient)
        base = _line_base(
            recipe_id=int(row.recipe_id),
            ingredient_idx=int(row.ingredient_idx),
            ingredient=ingredient_raw,
        )

        library = parse_ingredient_fields(ingredient_raw)
        library = finalize_parse_result(
            {**library, "confidence": None, "measurable": None, "final_method": None},
            ingredient_raw=ingredient_raw,
            final_method="library",
        )

        rules = rule_parse_fields(ingredient_raw)
        final = library
        final_method = "library"
        llm_row: dict[str, Any] | None = None
        llm_called = False

        if should_force_unmeasurable(library, ingredient_raw=ingredient_raw):
            final = finalize_parse_result(
                {**library, "measurable": False, "parse_status": "unmeasurable"},
                ingredient_raw=ingredient_raw,
                final_method="library",
            )
        elif not needs_llm_fallback(library, ingredient_raw=ingredient_raw):
            final = finalize_parse_result(library, ingredient_raw=ingredient_raw, final_method="library")
        elif rules_accepted(rules, threshold=rules_threshold):
            merged = merge_parse_result(library, rules, final_method="rule")
            final = finalize_parse_result(merged, ingredient_raw=ingredient_raw, final_method="rule")
            final_method = "rule"
        elif use_llm:
            merged = merge_parse_result(library, rules, final_method="rule")
            final, llm_row = _apply_llm(ingredient_raw, merged, llm_cache)
            llm_called = llm_row is not None
            final_method = final.get("final_method", "llm")
        else:
            if rules_accepted(rules, threshold=0.0):
                merged = merge_parse_result(library, rules, final_method="rule")
                final = finalize_parse_result(merged, ingredient_raw=ingredient_raw, final_method="rule")
                final_method = "rule"
            else:
                final = finalize_parse_result(library, ingredient_raw=ingredient_raw, final_method="library")

        record = {
            **base,
            **_prefix_fields(library, "library"),
            **_prefix_fields(rules, "rules"),
            "llm_called": llm_called,
            "llm_certainty": llm_row.get("llm_certainty") if llm_row else None,
            "llm_rationale": llm_row.get("llm_rationale") if llm_row else None,
            "llm_error": llm_row.get("llm_error") if llm_row else None,
            **_prefix_fields(llm_row["fields"] if llm_row else {f: None for f in PARSE_FIELDS}, "llm"),
            **_prefix_fields(final, "final"),
            "final_method": final_method,
            "pipeline_version": "lib_rules_llm",
        }
        rows.append(record)

    results_df = pd.DataFrame(rows)
    llm_calls_df = pd.DataFrame(llm_call_rows) if llm_call_rows else pd.DataFrame()
    meta = {
        "pipeline_version": "lib_rules_llm",
        "use_llm": use_llm,
        "rules_confidence_threshold": rules_threshold,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_llm_calls": len(llm_calls_df),
    }
    return results_df, llm_calls_df, meta


def compute_pipeline_report(
    results_df: pd.DataFrame,
    llm_calls_df: pd.DataFrame,
    *,
    model: str,
    meta: dict[str, Any],
    n_recipes: int,
    seed: int,
    rules_threshold: float = RULES_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate metrics for MLflow and local summary JSON."""
    from ingredient_parse_llm import pricing_for_model

    n_lines = len(results_df)
    n_unique = int(results_df["ingredient_norm"].nunique()) if n_lines else 0
    pricing = pricing_for_model(model)

    prompt_total = int(llm_calls_df["prompt_tokens"].sum()) if not llm_calls_df.empty else 0
    completion_total = int(llm_calls_df["completion_tokens"].sum()) if not llm_calls_df.empty else 0
    cost_input = prompt_total / 1e6 * pricing["input"]
    cost_output = completion_total / 1e6 * pricing["output"]
    cost_total = cost_input + cost_output

    final_ok = results_df["final_parse_status"] == "ok"
    final_measurable = results_df["final_measurable"] == True  # noqa: E712
    has_qty = results_df["final_quantity"].notna()

    certainty = pd.to_numeric(
        results_df.loc[results_df["llm_called"], "llm_certainty"],
        errors="coerce",
    ).dropna()

    def q(series: pd.Series, p: float) -> float | None:
        return round(float(series.quantile(p)), 4) if len(series) else None

    status_counts = results_df["final_parse_status"].value_counts().to_dict()
    method_counts = results_df["final_method"].value_counts().to_dict()
    library_status_counts = results_df["library_parse_status"].value_counts().to_dict()

    n_llm_calls = int(meta.get("n_llm_calls", 0))
    n_llm_resolved = int(
        ((results_df["final_method"] == "llm") & final_ok).sum()
    )
    n_rules_resolved = int(
        ((results_df["final_method"] == "rule") & final_ok).sum()
    )
    n_library_only = int(
        ((results_df["final_method"] == "library") & final_ok).sum()
    )

    return {
        "pipeline_version": meta.get("pipeline_version"),
        "model": model,
        "seed": seed,
        "n_recipes": n_recipes,
        "n_ingredient_lines": n_lines,
        "n_unique_ingredients": n_unique,
        "prompt_version": "parse_v1",
        "rules_confidence_threshold": rules_threshold,
        "use_llm": meta.get("use_llm", True),
        "coverage_ok_rate": round(float(final_ok.mean()), 4) if n_lines else 0.0,
        "coverage_measurable_rate": round(
            float((final_measurable & has_qty).mean()), 4
        ) if n_lines else 0.0,
        "unmeasurable_rate": round(
            float((results_df["final_parse_status"] == "unmeasurable").mean()), 4
        ) if n_lines else 0.0,
        "library_only_rate": round(n_library_only / max(n_lines, 1), 4),
        "rules_resolved_rate": round(n_rules_resolved / max(n_lines, 1), 4),
        "llm_resolved_rate": round(n_llm_resolved / max(n_lines, 1), 4),
        "llm_call_rate": round(n_llm_calls / max(n_unique, 1), 4),
        "still_unparsed_rate": round(
            float((~has_qty & (results_df["final_parse_status"] != "unmeasurable")).mean()),
            4,
        ) if n_lines else 0.0,
        "llm_error_rate": round(
            float(results_df["llm_error"].notna().mean()), 4
        ) if n_lines else 0.0,
        "llm_certainty_mean": round(float(certainty.mean()), 4) if len(certainty) else None,
        "llm_certainty_median": q(certainty, 0.5),
        "llm_certainty_p90": q(certainty, 0.9),
        "n_llm_calls": n_llm_calls,
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": completion_total,
        "total_tokens": prompt_total + completion_total,
        "cost_input_usd": round(cost_input, 4),
        "cost_output_usd": round(cost_output, 4),
        "cost_total_usd": round(cost_total, 4),
        "cost_avg_per_unique_ingredient_usd": round(cost_total / max(n_unique, 1), 6),
        "elapsed_sec": meta.get("elapsed_sec"),
        "final_parse_status_counts": status_counts,
        "final_method_counts": method_counts,
        "library_parse_status_counts": library_status_counts,
        "pricing_per_1m": pricing,
    }
