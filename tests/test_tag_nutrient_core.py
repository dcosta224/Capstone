"""Tests for tag_nutrient_core."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tag_dimensions import NUTRIENT_DIMENSIONS
from tag_nutrient_core import (
    absolute_label,
    add_corpus_labels,
    build_recipe_nutrient_wide,
    corpus_label_from_percentile,
    nutrient_lookup_from_df,
    resolve_nutrient_amount,
)


def test_resolve_nutrient_fallback():
    dim = next(d for d in NUTRIENT_DIMENSIONS if d.slug == "added_sugars")
    lookup = {(1, 2000): 12.0}
    val, nid = resolve_nutrient_amount(1, dim, lookup)
    assert val == 12.0
    assert nid == 2000


def test_recipe_rollup_sums_contributions():
    ingredients = pd.DataFrame(
        [
            {"recipe_id": 1, "recipe_name": "Soup", "ingredient_idx": 0, "fdc_id": 10, "gram_weight": 100.0},
            {"recipe_id": 1, "recipe_name": "Soup", "ingredient_idx": 1, "fdc_id": 20, "gram_weight": 50.0},
        ]
    )
    fn = pd.DataFrame(
        [
            {"fdc_id": 10, "nutrient_id": 1093, "amount": 400.0},
            {"fdc_id": 20, "nutrient_id": 1093, "amount": 200.0},
        ]
    )
    lookup = nutrient_lookup_from_df(fn)
    dim_ids = {"sodium": 1}
    wide = build_recipe_nutrient_wide(ingredients, lookup, dim_ids)
    row = wide[wide["dimension_slug"] == "sodium"].iloc[0]
    assert row["absolute_per_serving"] == pytest.approx(400 + 100)


def test_corpus_labels():
    dim = next(d for d in NUTRIENT_DIMENSIONS if d.slug == "sodium")
    assert corpus_label_from_percentile(10, dim) == "low"
    assert corpus_label_from_percentile(90, dim) == "high"


def test_absolute_label_sodium():
    dim = next(d for d in NUTRIENT_DIMENSIONS if d.slug == "sodium")
    assert absolute_label(50, dim) == "low"
    assert absolute_label(500, dim) == "high"


def test_add_corpus_labels_percentiles():
    df = pd.DataFrame(
        [
            {
                "recipe_id": 1,
                "recipe_name": "A",
                "dimension_slug": "fiber",
                "dimension_id": 1,
                "absolute_total": 5,
                "absolute_per_serving": 5,
                "nutrient_id_used": 1079,
                "n_ingredients_with_value": 1,
                "total_gram_weight": 100,
            },
            {
                "recipe_id": 2,
                "recipe_name": "B",
                "dimension_slug": "fiber",
                "dimension_id": 1,
                "absolute_total": 15,
                "absolute_per_serving": 15,
                "nutrient_id_used": 1079,
                "n_ingredients_with_value": 1,
                "total_gram_weight": 100,
            },
        ]
    )
    out, refs = add_corpus_labels(df)
    assert len(refs) > 0
    assert out["corpus_percentile"].notna().all()
