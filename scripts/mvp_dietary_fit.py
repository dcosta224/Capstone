"""User-relative dietary fit scoring for MVP ranker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mvp_nutrient_fit import nutrient_fit_to_score, nutrient_range_fit
from tag_dimensions import DIMENSION_BY_SLUG
from diet_tags_core import load_diet_tags
from tag_restrictions_core import load_taxonomy, recipe_is_free


# Map legacy free_label values to diet_tags slugs.
_FREE_LABEL_TO_TAG = {
    "dairy_free": "dairy_free",
    "egg_free": "egg_free",
    "fish_free": "fish_free",
    "shellfish_free": "shellfish_free",
    "tree_nut_free": "nut_free",
    "peanut_free": "peanut_free",
    "soy_free": "soy_free",
    "gluten_free": "gluten_free",
    "sesame_free": "sesame_free",
    "meat_free": "plant_based",
}


@dataclass
class DietaryProfile:
    """Optional per-nutrient bounds (per recipe serving) and required diet tags."""

    min_protein_g: float | None = None
    max_protein_g: float | None = None
    min_calcium_mg: float | None = None
    max_sodium_mg: float | None = None
    min_fiber_g: float | None = None
    max_saturated_fat_g: float | None = None
    max_added_sugars_g: float | None = None
    required_free_labels: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    w_dietary: float = 0.25

    def nutrient_bounds(self) -> dict[str, tuple[float | None, float | None]]:
        return {
            "protein": (self.min_protein_g, self.max_protein_g),
            "calcium": (self.min_calcium_mg, None),
            "sodium": (None, self.max_sodium_mg),
            "fiber": (self.min_fiber_g, None),
            "saturated_fat": (None, self.max_saturated_fat_g),
            "added_sugars": (None, self.max_added_sugars_g),
        }


def _recipe_nutrient_value(row: dict[str, Any], slug: str) -> float | None:
    col = f"tag_{slug}"
    if col in row and row[col] is not None:
        v = float(row[col])
        return v if math.isfinite(v) else None
    alt = {
        "protein": "protein_g",
        "saturated_fat": "fatty_acids_total_saturated_g",
        "fiber": "fiber_total_dietary_g",
        "sodium": "sodium_na_mg",
        "calcium": "calcium_ca_mg",
        "added_sugars": "sugars_added_g",
    }.get(slug)
    if alt and alt in row and row[alt] is not None:
        v = float(row[alt])
        return v if math.isfinite(v) else None
    return None


def dietary_range_fit(value: float, low: float | None, high: float | None) -> float:
    if low is None and high is None:
        return 0.0
    lo = low if low is not None else 0.0
    hi = high if high is not None else float("inf")
    if hi <= 0 and low is None:
        hi = float("inf")
    p = np.array([max(value, 1e-9)])
    L = np.array([max(lo, 1e-9)])
    U = np.array([hi if math.isfinite(hi) else 1e12])
    return nutrient_range_fit(p, L, U)


def dietary_fit_for_recipe(
    nutrient_row: dict[str, Any],
    profile: DietaryProfile,
    *,
    recipe_restrictions: set[str] | None = None,
    recipe_tags: dict[str, bool | None] | None = None,
    rules: dict | None = None,
) -> tuple[float, bool, list[str]]:
    """
    Return (fit_score 0–1, passes_hard_filters, violation_messages).
    Hard filter: required tags / free labels violated → score 0, passes False.
    """
    if rules is None:
        rules, _ = load_taxonomy()

    violations: list[str] = []
    recipe_restrictions = recipe_restrictions or set()
    recipe_tags = recipe_tags or {}

    for free_label in profile.required_free_labels:
        if not recipe_is_free(recipe_restrictions, free_label, rules):
            violations.append(f"contains {free_label.replace('_', ' ')}")
    for tag_slug in profile.required_tags:
        val = recipe_tags.get(tag_slug)
        if val is False:
            violations.append(f"fails required tag {tag_slug}")
        elif val is None:
            col = f"tag_{tag_slug}"
            if col in nutrient_row and nutrient_row[col] is False:
                violations.append(f"fails required tag {tag_slug}")
    for free_label in profile.required_free_labels:
        mapped = _FREE_LABEL_TO_TAG.get(free_label)
        if mapped and recipe_tags.get(mapped) is False:
            violations.append(f"fails {mapped}")
    if violations:
        return 0.0, False, violations

    fits: list[float] = []
    in_range_flags: list[bool] = []
    for slug, (lo, hi) in profile.nutrient_bounds().items():
        if lo is None and hi is None:
            continue
        val = _recipe_nutrient_value(nutrient_row, slug)
        if val is None:
            continue
        fit = dietary_range_fit(val, lo, hi)
        in_range = fit <= 1e-6
        fits.append(fit)
        in_range_flags.append(in_range)
        if not in_range:
            dim = DIMENSION_BY_SLUG.get(slug)
            unit = dim.unit if dim else ""
            if hi is not None and val > hi:
                violations.append(f"{slug} {val:.1f}{unit} above max {hi:.1f}")
            if lo is not None and val < lo:
                violations.append(f"{slug} {val:.1f}{unit} below min {lo:.1f}")

    if not fits:
        return 1.0, True, violations

    scores = [
        nutrient_fit_to_score(f, in_range=ir) for f, ir in zip(fits, in_range_flags, strict=True)
    ]
    return float(np.mean(scores)), True, violations


def merge_dietary_score(
    semantic_score: float,
    nutrient_score: float,
    dietary_score: float,
    *,
    w_semantic: float,
    w_nutrient: float,
    w_dietary: float,
) -> float:
    parts = [
        (w_semantic, semantic_score),
        (w_nutrient, nutrient_score),
        (w_dietary, dietary_score),
    ]
    total_w = sum(w for w, _ in parts if w > 0)
    if total_w <= 0:
        return 100.0 * semantic_score
    return 100.0 * sum(w * s for w, s in parts if w > 0) / total_w
