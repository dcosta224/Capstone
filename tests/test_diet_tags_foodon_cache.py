"""Tests for FoodOn cache integration in diet_tags_core."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foodon_web"))

from diet_tags_core import detect_contains, load_diet_tags, tag_ingredient


def _mini_cache() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "foodon_id": "FOODON_CHEDDAR",
                "label": "cheddar cheese",
                "contains_dairy": True,
                "contains_poultry": False,
            },
            {
                "foodon_id": "FOODON_BROTH",
                "label": "chicken broth",
                "contains_dairy": False,
                "contains_poultry": True,
            },
        ]
    )


def test_detect_contains_uses_cache_for_linked_node():
    registry = load_diet_tags()
    hits = detect_contains(
        "Some obscure name",
        None,
        registry,
        foodon_node_id="FOODON_CHEDDAR",
        foodon_contains_table=_mini_cache(),
    )
    assert "dairy" in hits
    assert hits["dairy"]["source"] == "foodon_cache"


def test_keywords_union_cache():
    registry = load_diet_tags()
    result = tag_ingredient(
        1,
        "Chicken broth, low sodium",
        None,
        registry,
        foodon_node_id="FOODON_BROTH",
        foodon_contains_table=_mini_cache(),
    )
    assert "poultry" in result["contains_set"]
    assert result["contains"]["poultry"]["source"] in {"description", "foodon_cache"}
