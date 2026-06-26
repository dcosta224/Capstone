"""Tests for diet_tags_corpus."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diet_tags_corpus import build_recipe_diet_tags_for_corpus


def test_build_recipe_diet_tags_for_corpus():
    ingredients_by_recipe = {
        1: pd.DataFrame(
            [
                {
                    "fdc_id": 101,
                    "fdc_description": "Chicken breast, raw",
                    "gram_weight": 150.0,
                },
                {
                    "fdc_id": 102,
                    "fdc_description": "Spinach, raw",
                    "gram_weight": 50.0,
                },
            ]
        ),
        2: pd.DataFrame(
            [
                {
                    "fdc_id": 103,
                    "fdc_description": "Milk, whole",
                    "gram_weight": 200.0,
                },
            ]
        ),
    }
    food_nutrients = pd.DataFrame(
        [
            {"fdc_id": 102, "nutrient_id": 1005, "amount": 3.0},
        ]
    )

    tags, restrictions = build_recipe_diet_tags_for_corpus(
        ingredients_by_recipe,
        food_nutrients,
        use_foodon=False,
    )

    assert tags[1]["plant_based"] is False
    assert tags[1]["vegan"] is False
    assert "poultry" in restrictions[1]
    assert tags[2]["dairy_free"] is False
    assert tags[2]["vegan"] is False
    assert "milk" in restrictions[2]
