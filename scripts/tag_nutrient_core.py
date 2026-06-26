"""Compute ingredient and recipe nutrient tags with absolute and corpus-relative labels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tag_dimensions import (
    CORPUS_HIGH_PERCENTILE,
    CORPUS_LOW_PERCENTILE,
    NUTRIENT_DIMENSIONS,
    NutrientDimension,
)


def resolve_nutrient_amount(
    fdc_id: int,
    dim: NutrientDimension,
    nutrient_lookup: dict[tuple[int, int], float],
) -> tuple[float | None, int | None]:
    """Return (amount per 100g, nutrient_id used) with optional fallback."""
    val = nutrient_lookup.get((fdc_id, dim.nutrient_id))
    if val is not None:
        return val, dim.nutrient_id
    if dim.fallback_nutrient_id is not None:
        fb = nutrient_lookup.get((fdc_id, dim.fallback_nutrient_id))
        if fb is not None:
            return fb, dim.fallback_nutrient_id
    return None, None


def absolute_label(
    value: float,
    dim: NutrientDimension,
) -> str:
    low = dim.low_absolute_threshold
    high = dim.high_absolute_threshold
    if low is None or high is None:
        return "medium"
    if dim.direction == "lower_better":
        if value <= low:
            return "low"
        if value >= high:
            return "high"
    else:
        if value >= high:
            return "high"
        if value <= low:
            return "low"
    return "medium"


def corpus_label_from_percentile(
    percentile: float,
    dim: NutrientDimension,
) -> str:
    if dim.direction == "lower_better":
        if percentile <= CORPUS_LOW_PERCENTILE:
            return "low"
        if percentile >= CORPUS_HIGH_PERCENTILE:
            return "high"
    else:
        if percentile >= CORPUS_HIGH_PERCENTILE:
            return "high"
        if percentile <= CORPUS_LOW_PERCENTILE:
            return "low"
    return "medium"


def percentile_rank(values: np.ndarray, x: float) -> float:
    """Percentile of x within values (0–100), ignoring NaN."""
    clean = values[np.isfinite(values)]
    if clean.size == 0 or not math.isfinite(x):
        return float("nan")
    return float(100.0 * np.mean(clean <= x))


def build_ingredient_nutrient_rows(
    fdc_ids: list[int],
    nutrient_lookup: dict[tuple[int, int], float],
    dimension_ids: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fdc_id in fdc_ids:
        for dim in NUTRIENT_DIMENSIONS:
            amount, nid = resolve_nutrient_amount(fdc_id, dim, nutrient_lookup)
            if amount is None:
                continue
            rows.append(
                {
                    "fdc_id": fdc_id,
                    "dimension_id": dimension_ids[dim.slug],
                    "absolute_per_100g": amount,
                    "nutrient_id_used": nid,
                }
            )
    return rows


def build_recipe_nutrient_wide(
    ingredients: pd.DataFrame,
    nutrient_lookup: dict[tuple[int, int], float],
    dimension_ids: dict[str, int],
) -> pd.DataFrame:
    """Aggregate recipe totals per dimension slug."""
    records: list[dict[str, Any]] = []
    for recipe_id, grp in ingredients.groupby("recipe_id"):
        recipe_name = grp["recipe_name"].iloc[0]
        total_grams = float(grp["gram_weight"].sum())
        for dim in NUTRIENT_DIMENSIONS:
            contrib = 0.0
            n_with = 0
            nid_used: int | None = None
            for row in grp.itertuples(index=False):
                fdc_id = int(row.fdc_id)
                amount, nid = resolve_nutrient_amount(fdc_id, dim, nutrient_lookup)
                if amount is None:
                    continue
                contrib += amount * (float(row.gram_weight) / 100.0)
                n_with += 1
                nid_used = nid
            if n_with == 0:
                continue
            per_serving = contrib
            records.append(
                {
                    "recipe_id": int(recipe_id),
                    "recipe_name": recipe_name,
                    "dimension_slug": dim.slug,
                    "dimension_id": dimension_ids[dim.slug],
                    "absolute_total": contrib,
                    "absolute_per_serving": per_serving,
                    "nutrient_id_used": nid_used,
                    "n_ingredients_with_value": n_with,
                    "total_gram_weight": total_grams,
                }
            )
    return pd.DataFrame(records)


def add_corpus_labels(recipe_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Add corpus_percentile and corpus_label; return percentile reference rows."""
    if recipe_df.empty:
        return recipe_df, []

    out = recipe_df.copy()
    out["corpus_percentile"] = np.nan
    out["corpus_label"] = "medium"
    out["absolute_label"] = "medium"

    dim_by_slug = {d.slug: d for d in NUTRIENT_DIMENSIONS}
    percentile_refs: list[dict[str, Any]] = []

    for slug, dim in dim_by_slug.items():
        mask = out["dimension_slug"] == slug
        vals = out.loc[mask, "absolute_per_serving"].astype(float).to_numpy()
        if vals.size == 0:
            continue
        out.loc[mask, "corpus_percentile"] = [
            percentile_rank(vals, float(v)) for v in vals
        ]
        for pct in (10, 25, 50, 75, 90):
            percentile_refs.append(
                {
                    "dimension_id": int(out.loc[mask, "dimension_id"].iloc[0]),
                    "percentile": float(pct),
                    "value": float(np.nanpercentile(vals, pct)),
                    "n_recipes": int(vals.size),
                }
            )
        for idx in out.index[mask]:
            val = float(out.at[idx, "absolute_per_serving"])
            pct = float(out.at[idx, "corpus_percentile"])
            out.at[idx, "absolute_label"] = absolute_label(val, dim)
            if math.isfinite(pct):
                out.at[idx, "corpus_label"] = corpus_label_from_percentile(pct, dim)

    return out, percentile_refs


def nutrient_lookup_from_df(food_nutrients: pd.DataFrame) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    for row in food_nutrients.itertuples(index=False):
        lookup[(int(row.fdc_id), int(row.nutrient_id))] = float(row.amount)
    return lookup
