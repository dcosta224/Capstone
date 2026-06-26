"""Tests for tag_restrictions_core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tag_restrictions_core import (
    ingredient_restriction_rows,
    load_taxonomy,
    match_restrictions_in_text,
    recipe_is_free,
)


def test_milk_keyword_match():
    rules, _ = load_taxonomy()
    hits = match_restrictions_in_text("cheddar cheese, shredded", rules)
    assert any(slug == "milk" for slug, _ in hits)


def test_no_false_positive_on_salt():
    rules, universal = load_taxonomy()
    hits = match_restrictions_in_text("salt", rules, universal=universal)
    assert hits == []


def test_wheat_in_flour():
    rules, _ = load_taxonomy()
    hits = match_restrictions_in_text("all purpose flour", rules)
    assert any(slug == "wheat" for slug, _ in hits)


def test_recipe_gluten_free():
    rules, _ = load_taxonomy()
    assert recipe_is_free(set(), "gluten_free", rules)
    assert not recipe_is_free({"wheat"}, "gluten_free", rules)


def test_ingredient_rows_from_description():
    rules, universal = load_taxonomy()
    rows = ingredient_restriction_rows(42, "Chicken breast, raw", None, rules, universal=universal)
    slugs = {r["restriction_slug"] for r in rows}
    assert "poultry" in slugs


def test_foodon_milk_ancestor():
    from foodon_index import FoodOnIndex

    index = FoodOnIndex(
        {
            "labels": {
                "FOODON_03315150": "mammalian milk product",
                "FOODON_X": "cheddar cheese",
            },
            "parents": {"FOODON_03315150": [], "FOODON_X": ["FOODON_03315150"]},
            "children": {"FOODON_03315150": ["FOODON_X"], "FOODON_X": []},
            "roots": ["FOODON_03315150"],
            "descendant_counts": {"FOODON_03315150": 1, "FOODON_X": 0},
            "label_keys": ["FOODON_03315150", "FOODON_X"],
        }
    )
    index.best_match = lambda text, min_score=0.55: {  # type: ignore[method-assign]
        "id": "FOODON_X",
        "label": "fontina cheese",
        "score": 0.99,
        "descendant_count": 0,
    }
    rules, universal = load_taxonomy()
    rows = ingredient_restriction_rows(
        1,
        "Fontina, semi-soft",
        None,
        rules,
        universal=universal,
        foodon_index=index,
    )
    slugs = {r["restriction_slug"] for r in rows}
    assert "milk" in slugs
    assert any(r["source"] == "foodon" for r in rows)
