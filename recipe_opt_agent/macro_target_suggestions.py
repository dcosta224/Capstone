"""Suggest a macro target box from the neighborhood mean nutrient profile.

One playground preset:
  - neighborhood_mean: mean resolved PFC across all neighborhood recipes, with a
    ±5% box around each macro (rounded to the nearest percent).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_simplex(p: float, c: float, f: float) -> tuple[float, float, float]:
    p, c, f = max(0.0, float(p)), max(0.0, float(c)), max(0.0, float(f))
    s = p + c + f
    if s <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return p / s, c / s, f / s


def _neighborhood_mean_pfc(lines_df) -> tuple[float, float, float] | None:
    """Mean resolved PFC (calorie-share) across every neighborhood recipe."""
    from recipe_opt_agent.problem_loader import _batch_recipe_pfc_from_lines

    pfc_by_rid = _batch_recipe_pfc_from_lines(lines_df)
    if not pfc_by_rid:
        return None
    ps = [float(v.get("protein", 0.0)) for v in pfc_by_rid.values()]
    cs = [float(v.get("carbs", 0.0)) for v in pfc_by_rid.values()]
    fs = [float(v.get("fat", 0.0)) for v in pfc_by_rid.values()]
    return _normalize_simplex(float(np.mean(ps)), float(np.mean(cs)), float(np.mean(fs)))


def _rounded_box_pm(mid: tuple[float, float, float], pad_pct: int = 5) -> dict[str, float]:
    """±pad_pct box around each macro, midpoint rounded to nearest percent."""
    p, c, f = (round(x * 100) for x in mid)
    box_pct = {
        "protein_min": p - pad_pct,
        "protein_max": p + pad_pct,
        "carb_min": c - pad_pct,
        "carb_max": c + pad_pct,
        "fat_min": f - pad_pct,
        "fat_max": f + pad_pct,
    }
    return {k: max(0.0, min(1.0, v / 100.0)) for k, v in box_pct.items()}


def suggest_macro_targets(
    lines_df,
    starting_recipe_id: str | None = None,
    *,
    pad_pct: int = 5,
    **_ignored: Any,
) -> dict[str, Any]:
    """Return the neighborhood-mean ±pad_pct macro box."""
    mean_pfc = _neighborhood_mean_pfc(lines_df)
    if mean_pfc is None:
        return {"error": "no_neighborhood_pfc", "n_recipes": 0, "presets": {}}

    from recipe_opt_agent.problem_loader import _batch_recipe_pfc_from_lines

    n_recipes = len(_batch_recipe_pfc_from_lines(lines_df))
    mean_box = _rounded_box_pm(mean_pfc, pad_pct=pad_pct)
    presets = {
        "neighborhood_mean": {
            "kind": "neighborhood_mean",
            "label": (
                f"Mean nutrient profile of the neighborhood, with a ±{pad_pct}% box "
                "around each macro (rounded to the nearest percent)."
            ),
            "box": mean_box,
            "midpoint": {
                "protein": mean_pfc[0],
                "carbs": mean_pfc[1],
                "fat": mean_pfc[2],
            },
            "n_recipes": n_recipes,
            "pad_pct": pad_pct,
        }
    }
    return {
        "n_recipes": n_recipes,
        "starting_recipe_id": str(starting_recipe_id) if starting_recipe_id else None,
        "neighborhood_mean_pfc": {
            "protein": mean_pfc[0],
            "carbs": mean_pfc[1],
            "fat": mean_pfc[2],
        },
        "presets": presets,
        "note": (
            "Mean resolved PFC across all Jaccard neighborhood recipes; "
            f"box is ±{pad_pct}% on each macro after rounding the mean to the nearest percent."
        ),
    }


def suggest_macro_targets_for_canonical(
    canonical_id: int,
    *,
    half_widths: dict[str, float] | None = None,
    fast_neighborhood: bool = True,
) -> dict[str, Any]:
    from canonical_optimization import CanonicalNeighborhood

    nb = CanonicalNeighborhood.build(
        int(canonical_id),
        fast=fast_neighborhood,
        use_cache=True,
    )
    # half_widths unused; kept for API compatibility with the server request model.
    _ = half_widths
    result = suggest_macro_targets(
        nb.lines_df,
        str(nb.starting_recipe_id),
        pad_pct=5,
    )
    result["canonical_id"] = int(canonical_id)
    result["title"] = getattr(nb, "title", None) or getattr(nb, "canonical_title", None)
    result["neighborhood_from_cache"] = bool(getattr(nb, "from_cache", False))
    result["n_neighborhood_recipes"] = len(nb.recipe_ids or [])
    return result


def high_protein_targets_from_mean(
    mean_pfc: tuple[float, float, float],
    *,
    protein_delta: float = 0.10,
    carb_delta: float = -0.05,
    fat_delta: float = -0.05,
    pad_pct: int = 2,
) -> dict[str, Any]:
    """Shift neighborhood mean PFC toward higher protein, then ±pad_pct box.

    Default: protein +10pp, carbs −5pp, fat −5pp (sum preserved), then
    renormalize onto the simplex and round midpoint to the nearest percent.
    """
    p0, c0, f0 = mean_pfc
    raw = (p0 + protein_delta, c0 + carb_delta, f0 + fat_delta)
    mid = _normalize_simplex(*raw)
    # Clamp each mid so ±pad leaves a non-empty [0,1] interval after rounding.
    mid_pct = [round(x * 100) for x in mid]
    # Re-normalize rounded percents so they sum to 100 when possible.
    s = sum(mid_pct)
    if s > 0 and s != 100:
        # Adjust the largest component so percents sum to 100.
        adj = 100 - s
        i_max = int(np.argmax(mid_pct))
        mid_pct[i_max] = max(pad_pct, min(100 - 2 * pad_pct, mid_pct[i_max] + adj))
    mid = (mid_pct[0] / 100.0, mid_pct[1] / 100.0, mid_pct[2] / 100.0)
    box = _rounded_box_pm(mid, pad_pct=pad_pct)
    return {
        "midpoint": {"protein": mid[0], "carbs": mid[1], "fat": mid[2]},
        "box": box,
        "protein_delta": protein_delta,
        "carb_delta": carb_delta,
        "fat_delta": fat_delta,
        "pad_pct": pad_pct,
        "neighborhood_mean_pfc": {
            "protein": float(p0),
            "carbs": float(c0),
            "fat": float(f0),
        },
    }


def suggest_high_protein_targets_for_canonical(
    canonical_id: int,
    *,
    protein_delta: float = 0.10,
    carb_delta: float = -0.05,
    fat_delta: float = -0.05,
    pad_pct: int = 2,
    fast_neighborhood: bool = True,
) -> dict[str, Any]:
    """Neighborhood-mean PFC → high-protein target box for one canonical dish."""
    from canonical_optimization import CanonicalNeighborhood

    nb = CanonicalNeighborhood.build(
        int(canonical_id),
        fast=fast_neighborhood,
        use_cache=True,
    )
    mean_pfc = _neighborhood_mean_pfc(nb.lines_df)
    if mean_pfc is None:
        return {"error": "no_neighborhood_pfc", "canonical_id": int(canonical_id)}
    hp = high_protein_targets_from_mean(
        mean_pfc,
        protein_delta=protein_delta,
        carb_delta=carb_delta,
        fat_delta=fat_delta,
        pad_pct=pad_pct,
    )
    return {
        "canonical_id": int(canonical_id),
        "title": getattr(nb, "title", None),
        "starting_recipe_id": str(nb.starting_recipe_id),
        "n_neighborhood_recipes": len(nb.recipe_ids or []),
        "neighborhood_from_cache": bool(getattr(nb, "from_cache", False)),
        **hp,
        "note": (
            f"Protein +{protein_delta:.0%} / carbs {carb_delta:+.0%} / fat {fat_delta:+.0%} "
            f"vs neighborhood mean PFC, then ±{pad_pct}% box (rounded)."
        ),
    }
