"""Build recipe diet tags from MVP corpus ingredients (no pre-built parquet required)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from diet_tags_core import load_diet_tags, tag_ingredient, tag_recipe

# Map diet_tags contains slugs to allergen_taxonomy restriction keys.
CONTAINS_TO_RESTRICTION: dict[str, str] = {
    "dairy": "milk",
    "egg": "eggs",
    "fish": "fish",
    "shellfish": "shellfish",
    "peanut": "peanuts",
    "tree_nut": "tree_nuts",
    "soy": "soybeans",
    "wheat": "wheat",
    "sesame": "sesame",
    "red_meat": "meat",
    "pork": "meat",
    "poultry": "poultry",
}


def nutrient_lookup_from_food_df(food_nutrients: pd.DataFrame) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    if food_nutrients is None or food_nutrients.empty:
        return lookup
    for row in food_nutrients.itertuples(index=False):
        if pd.notna(row.amount):
            lookup[(int(row.fdc_id), int(row.nutrient_id))] = float(row.amount)
    return lookup


def _optional_foodon_index():
    try:
        from foodon_paths import FOODON_INDEX_CACHE
        from foodon_index import FoodOnIndex

        if FOODON_INDEX_CACHE.is_file():
            return FoodOnIndex.from_cache(FOODON_INDEX_CACHE)
        return FoodOnIndex.from_owl()
    except Exception:
        return None


def _recipe_nutrient_totals(
    lines: pd.DataFrame,
    nutrient_lookup: dict[tuple[int, int], float],
    registry,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in lines.itertuples(index=False):
        if row.fdc_id is None or pd.isna(row.fdc_id):
            continue
        grams = float(row.gram_weight) if row.gram_weight is not None and pd.notna(row.gram_weight) else 0.0
        if grams <= 0:
            continue
        fdc_id = int(row.fdc_id)
        for slug, spec in registry.nutrients.items():
            val = nutrient_lookup.get((fdc_id, spec.nutrient_id))
            if val is None and spec.fallback_nutrient_id is not None:
                val = nutrient_lookup.get((fdc_id, spec.fallback_nutrient_id))
            if val is None:
                continue
            totals[slug] = totals.get(slug, 0.0) + val * (grams / 100.0)
    return totals


def _contains_to_restrictions(contains_union: set[str]) -> set[str]:
    out: set[str] = set()
    for slug in contains_union:
        mapped = CONTAINS_TO_RESTRICTION.get(slug)
        if mapped:
            out.add(mapped)
    return out


def build_recipe_diet_tags_for_corpus(
    ingredients_by_recipe: dict[int, pd.DataFrame],
    food_nutrients: pd.DataFrame,
    *,
    use_foodon: bool = True,
) -> tuple[dict[int, dict[str, bool | None]], dict[int, set[str]]]:
    """
    Tag each recipe in an MVP-style corpus from resolved ingredient lines.

    Returns (recipe_diet_tags, recipe_restrictions) keyed by recipe_id.
    """
    registry = load_diet_tags()
    nutrient_lookup = nutrient_lookup_from_food_df(food_nutrients)
    foodon_index = _optional_foodon_index() if use_foodon else None

    fdc_meta: dict[int, tuple[str, str | None]] = {}
    for ing_df in ingredients_by_recipe.values():
        for row in ing_df.itertuples(index=False):
            if row.fdc_id is None or pd.isna(row.fdc_id):
                continue
            fdc_id = int(row.fdc_id)
            if fdc_id in fdc_meta:
                continue
            desc = str(row.fdc_description) if row.fdc_description is not None and pd.notna(row.fdc_description) else ""
            fdc_meta[fdc_id] = (desc, None)

    ingredient_cache: dict[int, dict[str, Any]] = {}
    for fdc_id, (desc, ing_text) in fdc_meta.items():
        tagged = tag_ingredient(
            fdc_id,
            desc,
            ing_text,
            registry,
            nutrient_lookup=nutrient_lookup,
            foodon_index=foodon_index,
        )
        ingredient_cache[fdc_id] = {
            "contains_set": set(tagged["contains_set"]),
            "tags": tagged["tags"],
        }

    recipe_diet_tags: dict[int, dict[str, bool | None]] = {}
    recipe_restrictions: dict[int, set[str]] = {}

    for recipe_id, ing_df in ingredients_by_recipe.items():
        rid = int(recipe_id)
        ing_rows: list[dict[str, Any]] = []
        contains_union: set[str] = set()
        for row in ing_df.itertuples(index=False):
            if row.fdc_id is None or pd.isna(row.fdc_id):
                ing_rows.append({"contains_set": set(), "tags": {}})
                continue
            cached = ingredient_cache.get(int(row.fdc_id))
            if cached is None:
                ing_rows.append({"contains_set": set(), "tags": {}})
                continue
            ing_rows.append(cached)
            contains_union |= set(cached["contains_set"])

        totals = _recipe_nutrient_totals(ing_df, nutrient_lookup, registry)
        result = tag_recipe(
            rid,
            str(recipe_id),
            ing_rows,
            registry,
            nutrient_totals_per_serving=totals,
        )
        recipe_diet_tags[rid] = dict(result["tags"])
        recipe_restrictions[rid] = _contains_to_restrictions(contains_union)

    return recipe_diet_tags, recipe_restrictions
