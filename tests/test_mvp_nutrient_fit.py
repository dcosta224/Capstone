import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mvp_nutrient_fit import (
    FRAC_EPS,
    clamp_fraction,
    nutrient_range_fit,
    nutrient_range_fit_from_totals,
    pfc_calorie_fractions,
)


def test_in_range_score_is_zero():
    p = np.array([0.30, 0.50, 0.20])
    L = np.array([0.25, 0.45, 0.15])
    U = np.array([0.35, 0.55, 0.25])
    assert nutrient_range_fit(p, L, U) == pytest.approx(0.0)


def test_zero_lower_bound_uses_epsilon_not_inf():
    score = nutrient_range_fit_from_totals(
        protein_g=30,
        fat_g=20,
        carbs_g=50,
        energy_kcal=500,
        fat_frac_min=0.0,
        fat_frac_max=0.35,
        carb_frac_min=0.30,
        carb_frac_max=0.90,
        protein_frac_min=0.10,
        protein_frac_max=0.60,
    )
    assert np.isfinite(score)
    assert score < 1.0


def test_clamp_fraction_maps_zero():
    assert clamp_fraction(0.0) == FRAC_EPS


def test_pfc_fractions_sum_near_one():
    fat_g, carbs_g, protein_g = 20.0, 50.0, 30.0
    energy_kcal = fat_g * 9 + carbs_g * 4 + protein_g * 4
    p = pfc_calorie_fractions(fat_g, carbs_g, protein_g, energy_kcal)
    assert p.sum() == pytest.approx(1.0, rel=0.01)
