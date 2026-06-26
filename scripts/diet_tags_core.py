"""Load diet_tags.json and evaluate ingredient/recipe tags."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIET_TAGS_PATH = ROOT / "data" / "diet_tags.json"
_WORD_BOUNDARY = r"\b{}\b"


@dataclass(frozen=True)
class ContainsTrigger:
    slug: str
    keywords: tuple[str, ...]
    foodon_ancestors: tuple[str, ...]


@dataclass(frozen=True)
class NutrientSpec:
    slug: str
    nutrient_id: int
    unit: str
    fallback_nutrient_id: int | None = None


@dataclass
class DietTagsRegistry:
    universal: frozenset[str]
    contains: dict[str, ContainsTrigger]
    nutrients: dict[str, NutrientSpec]
    ingredient_tags: dict[str, dict[str, Any]]
    recipe_tags: dict[str, dict[str, Any]]


def load_diet_tags(path: Path | None = None) -> DietTagsRegistry:
    p = path or DEFAULT_DIET_TAGS_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    contains = {
        slug: ContainsTrigger(
            slug=slug,
            keywords=tuple(str(k).lower() for k in spec.get("keywords", [])),
            foodon_ancestors=tuple(spec.get("foodon_ancestors", [])),
        )
        for slug, spec in raw["contains"].items()
    }
    nutrients = {}
    for slug, spec in raw["nutrients"].items():
        nutrients[slug] = NutrientSpec(
            slug=slug,
            nutrient_id=int(spec["nutrient_id"]),
            unit=str(spec["unit"]),
            fallback_nutrient_id=int(spec["fallback_nutrient_id"])
            if spec.get("fallback_nutrient_id") is not None
            else None,
        )
    return DietTagsRegistry(
        universal=frozenset(str(x).lower() for x in raw.get("universal_ingredients", [])),
        contains=contains,
        nutrients=nutrients,
        ingredient_tags=raw["ingredient_tags"],
        recipe_tags=raw["recipe_tags"],
    )


def _compile_patterns(keywords: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(_WORD_BOUNDARY.format(re.escape(kw)), re.IGNORECASE) for kw in keywords]


def detect_contains(
    description: str,
    ingredients_text: str | None,
    registry: DietTagsRegistry,
    *,
    foodon_index: Any | None = None,
    foodon_min_score: float = 0.55,
) -> dict[str, dict[str, str]]:
    """Return contains_slug -> {source, matched_term}."""
    hits: dict[str, dict[str, str]] = {}

    def _try_text(source: str, text: str) -> None:
        if not text or not text.strip():
            return
        normalized = text.lower().strip()
        if normalized in registry.universal:
            return
        for trigger in registry.contains.values():
            if trigger.slug in hits:
                continue
            for pat in _compile_patterns(trigger.keywords):
                m = pat.search(normalized)
                if m:
                    hits[trigger.slug] = {"source": source, "matched_term": m.group(0).lower()}
                    break

    _try_text("description", description)
    if ingredients_text:
        _try_text("ingredients", ingredients_text)

    if foodon_index is not None and description.strip():
        match = foodon_index.best_match(description, min_score=foodon_min_score)
        if match is not None:
            node_id = str(match["id"])
            for trigger in registry.contains.values():
                if trigger.slug in hits or not trigger.foodon_ancestors:
                    continue
                if foodon_index.matches_any_ancestor(node_id, trigger.foodon_ancestors):
                    hits[trigger.slug] = {"source": "foodon", "matched_term": node_id}

    return hits


def nutrient_values_per_100g(
    nutrient_lookup: dict[tuple[int, int], float],
    registry: DietTagsRegistry,
    fdc_id: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for slug, spec in registry.nutrients.items():
        val = nutrient_lookup.get((fdc_id, spec.nutrient_id))
        if val is None and spec.fallback_nutrient_id is not None:
            val = nutrient_lookup.get((fdc_id, spec.fallback_nutrient_id))
        if val is not None:
            out[slug] = float(val)
    return out


def _passes_nutrient_bounds(
    values: dict[str, float],
    mins: dict[str, float] | None,
    maxs: dict[str, float] | None,
) -> bool:
    if mins:
        for key, bound in mins.items():
            val = values.get(key)
            if val is None or val < bound:
                return False
    if maxs:
        for key, bound in maxs.items():
            val = values.get(key)
            if val is None or val > bound:
                return False
    return True


def _eval_ingredient_tag(
    tag_slug: str,
    spec: dict[str, Any],
    contains: set[str],
    nutrients: dict[str, float],
) -> bool | None:
    not_contains = spec.get("not_contains")
    if not_contains:
        blocked = not_contains if isinstance(not_contains, list) else [not_contains]
        if any(c in contains for c in blocked):
            return False
        return True

    mins = spec.get("nutrient_min")
    maxs = spec.get("nutrient_max")
    if mins or maxs:
        if not nutrients:
            return None
        return _passes_nutrient_bounds(nutrients, mins, maxs)

    return None


def tag_ingredient(
    fdc_id: int,
    description: str,
    ingredients_text: str | None,
    registry: DietTagsRegistry,
    *,
    nutrient_lookup: dict[tuple[int, int], float] | None = None,
    foodon_index: Any | None = None,
    foodon_min_score: float = 0.55,
) -> dict[str, Any]:
    """Return contains map, nutrients per 100g, and user-facing ingredient tag booleans."""
    contains_hits = detect_contains(
        description,
        ingredients_text,
        registry,
        foodon_index=foodon_index,
        foodon_min_score=foodon_min_score,
    )
    contains_set = set(contains_hits.keys())
    nutrients = nutrient_values_per_100g(nutrient_lookup or {}, registry, fdc_id)

    tag_values: dict[str, bool | None] = {}
    for tag_slug, spec in registry.ingredient_tags.items():
        tag_values[tag_slug] = _eval_ingredient_tag(tag_slug, spec, contains_set, nutrients)

    return {
        "fdc_id": fdc_id,
        "description": description,
        "contains": contains_hits,
        "contains_set": contains_set,
        "nutrients_per_100g": nutrients,
        "tags": tag_values,
    }


def _ingredient_passes_tag(
    tag_slug: str,
    ingredient_tag_rows: list[dict[str, Any]],
    registry: DietTagsRegistry,
) -> bool:
    spec = registry.ingredient_tags.get(tag_slug)
    if spec is None:
        return True
    for row in ingredient_tag_rows:
        val = row.get("tags", {}).get(tag_slug)
        if val is False:
            return False
        if val is None and spec.get("not_contains"):
            return False
    return True


def tag_recipe(
    recipe_id: int,
    recipe_name: str,
    ingredient_rows: list[dict[str, Any]],
    registry: DietTagsRegistry,
    *,
    nutrient_totals_per_serving: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Roll up ingredient tags + optional recipe nutrient totals."""
    contains_union: set[str] = set()
    for row in ingredient_rows:
        contains_union |= set(row.get("contains_set", set()))

    recipe_tags: dict[str, bool | None] = {}
    for tag_slug, spec in registry.recipe_tags.items():
        req = spec.get("requires_all_ingredients")
        if req:
            req_list = req if isinstance(req, list) else [req]
            if not all(_ingredient_passes_tag(t, ingredient_rows, registry) for t in req_list):
                recipe_tags[tag_slug] = False
                continue

        pair_rules = spec.get("forbid_contains_pair") or []
        pair_fail = False
        for pair in pair_rules:
            if all(p in contains_union for p in pair):
                pair_fail = True
                break
        if pair_fail:
            recipe_tags[tag_slug] = False
            continue

        not_any = spec.get("not_contains_any_ingredient") or []
        if not_any and any(c in contains_union for c in not_any):
            recipe_tags[tag_slug] = False
            continue

        mins = spec.get("nutrient_min_per_serving")
        maxs = spec.get("nutrient_max_per_serving")
        if mins or maxs:
            if not nutrient_totals_per_serving:
                recipe_tags[tag_slug] = None
            else:
                recipe_tags[tag_slug] = _passes_nutrient_bounds(
                    nutrient_totals_per_serving, mins, maxs
                )
            continue

        if req or pair_rules or not_any:
            recipe_tags[tag_slug] = True
        else:
            recipe_tags[tag_slug] = None

    return {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "contains_union": sorted(contains_union),
        "tags": recipe_tags,
        "nutrients_per_serving": nutrient_totals_per_serving or {},
    }


def flatten_ingredient_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Long-format rows for parquet export."""
    rows: list[dict[str, Any]] = []
    fdc_id = int(result["fdc_id"])
    for cslug, meta in result["contains"].items():
        rows.append(
            {
                "fdc_id": fdc_id,
                "row_type": "contains",
                "tag_slug": cslug,
                "value": True,
                "source": meta.get("source"),
                "matched_term": meta.get("matched_term"),
            }
        )
    for tslug, val in result["tags"].items():
        if val is None:
            continue
        rows.append(
            {
                "fdc_id": fdc_id,
                "row_type": "tag",
                "tag_slug": tslug,
                "value": bool(val),
                "source": "rule",
                "matched_term": None,
            }
        )
    for nslug, amount in result["nutrients_per_100g"].items():
        rows.append(
            {
                "fdc_id": fdc_id,
                "row_type": "nutrient",
                "tag_slug": nslug,
                "value": float(amount),
                "source": "usda",
                "matched_term": None,
            }
        )
    return rows
