"""Parse RecipeNLG `directions` lists and select steps relevant to an ingredient.

The matching pipeline only needs the recipe steps that actually mention the
ingredient being matched, so the LLM judge sees relevant context without paying
for the whole instruction block. Directions are stored as a pseudo-JSON list
string (same shape as `ingredients`): ``["step one", "step two", ...]``.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Units / measurement words that are useless as ingredient match tokens.
_STEP_FILTER_STOPWORDS = frozenset(
    """
    cup cups c tsp teaspoon teaspoons tbsp tablespoon tablespoons oz ounce ounces
    lb lbs pound pounds g gram grams kg ml l liter liters pkg package packages can
    cans jar jars bottle box bag bags pint pints quart quarts gallon stick sticks
    slice slices piece pieces dash pinch clove cloves and the for with into from
    add mix stir until about over more all your you can then
    """.split()
)


def parse_directions_list(value: Any) -> list[str]:
    """Itemize a RecipeNLG directions string into individual steps.

    Mirrors `ingredient_query_cache._parse_ingredient_list`: the field is a
    pseudo-JSON list whose elements are double-quoted and joined by ``", "``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    text = text.strip()
    if not text:
        return []
    if '", "' not in text:
        item = text.strip('"').strip()
        return [item] if item else []
    parts = text.split('", "')
    out: list[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if i == 0:
            part = part.removeprefix('["').removeprefix('"')
        if i == len(parts) - 1:
            part = part.removesuffix('"]').removesuffix('"')
        part = part.strip()
        if part:
            out.append(part)
    return out


def ingredient_match_tokens(ingredient: str, *, min_len: int = 3) -> set[str]:
    """Lowercased content tokens from a raw ingredient line for step matching.

    Drops short tokens, pure numbers, and measurement/stopwords so that a step
    is only kept when it shares a meaningful food word with the ingredient.
    """
    tokens: set[str] = set()
    for raw in re.split(r"[^a-z0-9]+", str(ingredient).lower()):
        if not raw or len(raw) < min_len:
            continue
        if raw.isdigit():
            continue
        if raw in _STEP_FILTER_STOPWORDS:
            continue
        tokens.add(raw)
    return tokens


def relevant_direction_steps(
    ingredient: str,
    steps: list[str],
    *,
    max_steps: int = 4,
) -> list[str]:
    """Steps mentioning any ingredient token, ranked by token-overlap count.

    Matching is substring-based on the lowercased step text so morphological
    variants ("chop"/"chopped") and casing exceptions are still caught. Returns
    at most `max_steps` steps, ordered by descending overlap then original order.
    """
    if not steps:
        return []
    tokens = ingredient_match_tokens(ingredient)
    if not tokens:
        return []

    scored: list[tuple[int, int, str]] = []
    for order, step in enumerate(steps):
        low = step.lower()
        overlap = sum(1 for tok in tokens if tok in low)
        if overlap > 0:
            scored.append((overlap, order, step))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [step for _, _, step in scored[:max_steps]]
