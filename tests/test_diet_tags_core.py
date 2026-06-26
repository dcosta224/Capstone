"""Tests for diet_tags_core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foodon_web"))

from diet_tags_core import load_diet_tags, tag_ingredient, tag_recipe


def _mock_index():
    from foodon_index import FoodOnIndex

    return FoodOnIndex(
        {
            "labels": {
                "FOODON_03315150": "mammalian milk product",
                "FOODON_X": "fontina cheese",
                "FOODON_PORK": "pork meat food product",
            },
            "parents": {
                "FOODON_03315150": [],
                "FOODON_X": ["FOODON_03315150"],
                "FOODON_PORK": [],
            },
            "children": {
                "FOODON_03315150": ["FOODON_X"],
                "FOODON_X": [],
                "FOODON_PORK": [],
            },
            "roots": ["FOODON_03315150"],
            "descendant_counts": {"FOODON_03315150": 1, "FOODON_X": 0, "FOODON_PORK": 0},
            "label_keys": ["FOODON_03315150", "FOODON_X", "FOODON_PORK"],
        }
    )


def test_dairy_free_fontina_via_foodon():
    registry = load_diet_tags()
    index = _mock_index()
    index.best_match = lambda text, min_score=0.55: {  # type: ignore[method-assign]
        "id": "FOODON_X",
        "label": "fontina cheese",
        "score": 0.99,
        "descendant_count": 0,
    }
    result = tag_ingredient(1, "Fontina, semi-soft", None, registry, foodon_index=index)
    assert result["tags"]["dairy_free"] is False
    assert result["tags"]["vegan"] is False


def test_chicken_poultry_contains():
    registry = load_diet_tags()
    result = tag_ingredient(2, "Chicken breast, raw", None, registry, foodon_index=None)
    assert "poultry" in result["contains_set"]
    assert result["tags"]["plant_based"] is False


def test_recipe_vegan_rollup():
    registry = load_diet_tags()
    ing_a = {"contains_set": set(), "tags": {"vegan": True, "dairy_free": True}}
    ing_b = {"contains_set": {"dairy"}, "tags": {"vegan": False, "dairy_free": False}}
    result = tag_recipe(10, "Test", [ing_a, ing_b], registry)
    assert result["tags"]["vegan"] is False
    assert result["tags"]["dairy_free"] is False


def test_kosher_style_meat_dairy_pair():
    registry = load_diet_tags()
    ing_a = {"contains_set": {"red_meat"}, "tags": {"no_pork": True, "shellfish_free": True}}
    ing_b = {"contains_set": {"dairy"}, "tags": {"dairy_free": False}}
    result = tag_recipe(11, "Burger with cheese", [ing_a, ing_b], registry)
    assert result["tags"]["kosher_style"] is False


def test_low_carb_nutrient_ingredient():
    registry = load_diet_tags()
    lookup = {(3, 1005): 5.0}
    result = tag_ingredient(
        3,
        "Spinach, raw",
        None,
        registry,
        nutrient_lookup=lookup,
    )
    assert result["tags"]["low_carb"] is True
