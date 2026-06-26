"""Deterministic allergen and dietary-restriction tagging from taxonomy keywords."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY_PATH = ROOT / "data" / "allergen_taxonomy.json"

_WORD_BOUNDARY = r"\b{}\b"


@dataclass(frozen=True)
class RestrictionRule:
    slug: str
    free_label: str
    allergen: bool
    keywords: tuple[str, ...]
    foodon_ancestors: tuple[str, ...]


def load_taxonomy(path: Path | None = None) -> tuple[dict[str, RestrictionRule], frozenset[str]]:
    p = path or DEFAULT_TAXONOMY_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    rules: dict[str, RestrictionRule] = {}
    for slug, spec in raw["restrictions"].items():
        rules[slug] = RestrictionRule(
            slug=slug,
            free_label=str(spec["free_label"]),
            allergen=bool(spec.get("allergen", True)),
            keywords=tuple(str(k).lower() for k in spec.get("description_keywords", [])),
            foodon_ancestors=tuple(spec.get("foodon_ancestors", [])),
        )
    universal = frozenset(str(x).lower() for x in raw.get("universal_ingredients", []))
    return rules, universal


def _compile_patterns(keywords: tuple[str, ...]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for kw in keywords:
        escaped = re.escape(kw)
        patterns.append(re.compile(_WORD_BOUNDARY.format(escaped), re.IGNORECASE))
    return patterns


def match_restrictions_in_text(
    text: str,
    rules: dict[str, RestrictionRule],
    *,
    universal: frozenset[str] | None = None,
) -> list[tuple[str, str]]:
    """Return list of (restriction_slug, matched_keyword) for text hits."""
    if not text or not text.strip():
        return []
    normalized = text.lower().strip()
    if universal and normalized in universal:
        return []

    hits: list[tuple[str, str]] = []
    for rule in rules.values():
        for pat in _compile_patterns(rule.keywords):
            m = pat.search(normalized)
            if m:
                hits.append((rule.slug, m.group(0).lower()))
                break
    return hits


def match_restrictions_foodon(
    description: str,
    rules: dict[str, RestrictionRule],
    index: Any,
    *,
    min_score: float = 0.55,
) -> list[tuple[str, str]]:
    """Match restrictions via FoodOn class ancestry (requires FoodOnIndex)."""
    if not description or not description.strip():
        return []
    match = index.best_match(description, min_score=min_score)
    if match is None:
        return []

    node_id = str(match["id"])
    hits: list[tuple[str, str]] = []
    for rule in rules.values():
        if not rule.foodon_ancestors:
            continue
        if index.matches_any_ancestor(node_id, rule.foodon_ancestors):
            hits.append((rule.slug, node_id))
    return hits


def ingredient_restriction_rows(
    fdc_id: int,
    description: str,
    ingredients_text: str | None,
    rules: dict[str, RestrictionRule],
    *,
    universal: frozenset[str] | None = None,
    foodon_index: Any | None = None,
    foodon_min_score: float = 0.55,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, text in (("description", description), ("ingredients", ingredients_text or "")):
        if not text:
            continue
        for slug, term in match_restrictions_in_text(text, rules, universal=universal):
            if slug in seen:
                continue
            seen.add(slug)
            rows.append(
                {
                    "fdc_id": fdc_id,
                    "restriction_slug": slug,
                    "source": source,
                    "matched_term": term,
                }
            )
        if foodon_index is not None and source == "description":
            for slug, node_id in match_restrictions_foodon(
                text,
                rules,
                foodon_index,
                min_score=foodon_min_score,
            ):
                if slug in seen:
                    continue
                seen.add(slug)
                rows.append(
                    {
                        "fdc_id": fdc_id,
                        "restriction_slug": slug,
                        "source": "foodon",
                        "matched_term": node_id,
                    }
                )
    return rows


def recipe_contains_restriction(
    restriction_slugs: set[str],
    slug: str,
) -> bool:
    return slug in restriction_slugs


def recipe_is_free(
    recipe_restrictions: set[str],
    free_label: str,
    rules: dict[str, RestrictionRule],
) -> bool:
    """Recipe is X-free if no triggering restriction slug maps to that free_label."""
    triggering = {
        slug for slug in recipe_restrictions if rules[slug].free_label == free_label
    }
    return len(triggering) == 0


def free_labels_for_recipe(
    recipe_restrictions: set[str],
    rules: dict[str, RestrictionRule],
) -> set[str]:
    all_labels = {r.free_label for r in rules.values()}
    blocked: set[str] = set()
    for slug in recipe_restrictions:
        if slug in rules:
            blocked.add(rules[slug].free_label)
    return all_labels - blocked
