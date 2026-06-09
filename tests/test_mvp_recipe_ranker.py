import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mvp_nutrient_fit import nutrient_fit_to_score
from mvp_recipe_ranker import rank_recipes


def _fake_corpus():
    ids = [1, 2, 3]
    names = ["A", "B", "C"]
    embs = np.array(
        [
            [1.0, 0.0, 0.0] + [0.0] * 381,
            [0.9, 0.1, 0.0] + [0.0] * 381,
            [0.0, 1.0, 0.0] + [0.0] * 381,
        ],
        dtype=np.float32,
    )
    nutrients = [
        {
            "protein_g": 30,
            "total_lipid_fat_g": 20,
            "carbohydrate_by_difference_g": 50,
            "energy_kcal": 600,
        },
        {
            "protein_g": 25,
            "total_lipid_fat_g": 15,
            "carbohydrate_by_difference_g": 60,
            "energy_kcal": 500,
        },
        {
            "protein_g": 10,
            "total_lipid_fat_g": 40,
            "carbohydrate_by_difference_g": 30,
            "energy_kcal": 700,
        },
    ]
    return ids, names, embs, nutrients


def test_ranker_orders_by_combined_score():
    ids, names, embs, nutrients = _fake_corpus()
    query = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
    ranked = rank_recipes(
        ids,
        names,
        embs,
        query,
        nutrients,
        kcal_min=400,
        kcal_max=600,
        fat_frac_min=0.25,
        fat_frac_max=0.35,
        carb_frac_min=0.45,
        carb_frac_max=0.55,
        protein_frac_min=0.15,
        protein_frac_max=0.25,
    )
    assert ranked[0].rank == 1
    assert ranked[0].semantic_sim >= ranked[-1].semantic_sim
    assert all(0 <= r.combined_score <= 100.0 + 1e-9 for r in ranked)
    assert ranked[0].combined_score >= ranked[-1].combined_score


def test_perfect_combined_score_is_100():
    ids = [1]
    names = ["Perfect"]
    embs = np.array([[1.0, 0.0, 0.0] + [0.0] * 381], dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
    # PFC fractions 30% fat, 50% carb, 20% protein at 500 kcal
    nutrients = [
        {
            "protein_g": 25,
            "total_lipid_fat_g": 16.7,
            "carbohydrate_by_difference_g": 62.5,
            "energy_kcal": 500,
        }
    ]
    ranked = rank_recipes(
        ids,
        names,
        embs,
        query,
        nutrients,
        kcal_min=400,
        kcal_max=600,
        fat_frac_min=0.25,
        fat_frac_max=0.35,
        carb_frac_min=0.45,
        carb_frac_max=0.55,
        protein_frac_min=0.15,
        protein_frac_max=0.25,
    )
    assert ranked[0].semantic_score == pytest.approx(1.0)
    assert ranked[0].nutrient_score == pytest.approx(1.0)
    assert ranked[0].combined_score == pytest.approx(100.0)


def test_weighted_sum_nonzero_when_one_component_good():
    ids = [1, 2]
    names = ["A", "B"]
    embs = np.array(
        [
            [1.0, 0.0, 0.0] + [0.0] * 381,
            [0.0, 1.0, 0.0] + [0.0] * 381,
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
    nutrients = [
        {
            "protein_g": 25,
            "total_lipid_fat_g": 16.7,
            "carbohydrate_by_difference_g": 62.5,
            "energy_kcal": 500,
        },
        {
            "protein_g": 5,
            "total_lipid_fat_g": 60,
            "carbohydrate_by_difference_g": 10,
            "energy_kcal": 700,
        },
    ]
    ranked = rank_recipes(
        ids,
        names,
        embs,
        query,
        nutrients,
        kcal_min=400,
        kcal_max=600,
        fat_frac_min=0.25,
        fat_frac_max=0.35,
        carb_frac_min=0.45,
        carb_frac_max=0.55,
        protein_frac_min=0.15,
        protein_frac_max=0.25,
        w_semantic=0.5,
        w_nutrient=0.5,
    )
    best = next(r for r in ranked if r.recipe_id == 1)
    assert best.combined_score == pytest.approx(100.0)
    assert best.combined_score > 0
    worst = next(r for r in ranked if r.recipe_id == 2)
    assert worst.combined_score > 0
    assert worst.combined_score < best.combined_score


def test_nutrient_fit_to_score():
    assert nutrient_fit_to_score(0.0, in_range=True) == pytest.approx(1.0)
    assert nutrient_fit_to_score(2.0) == pytest.approx(1.0 / 3.0)
    assert nutrient_fit_to_score(float("inf")) == pytest.approx(0.0)
