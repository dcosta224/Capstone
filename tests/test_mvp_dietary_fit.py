"""Tests for mvp_dietary_fit."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mvp_dietary_fit import DietaryProfile, dietary_fit_for_recipe, merge_dietary_score


def test_dietary_fit_within_sodium_bound():
    profile = DietaryProfile(max_sodium_mg=500)
    row = {"tag_sodium": 400}
    score, passes, violations = dietary_fit_for_recipe(row, profile)
    assert passes
    assert score == pytest.approx(1.0)
    assert violations == []


def test_dietary_hard_filter_gluten():
    profile = DietaryProfile(required_free_labels=("gluten_free",))
    row = {}
    score, passes, violations = dietary_fit_for_recipe(
        row, profile, recipe_restrictions={"wheat"}
    )
    assert not passes
    assert score == 0.0
    assert violations


def test_merge_dietary_score_weights():
    combined = merge_dietary_score(1.0, 1.0, 0.0, w_semantic=0.5, w_nutrient=0.5, w_dietary=0.0)
    assert combined == pytest.approx(100.0)
