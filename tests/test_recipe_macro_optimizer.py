import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recipe_macro_optimizer import (
    IngredientConstraint,
    IngredientMeta,
    MacroBounds,
    OptimizerConfig,
    RecipeMacroOptimizer,
    build_macro_matrix,
    check_macro_region_feasible,
    compute_macros,
    display_portion_change_metrics,
    format_serving_display,
    kcal_scaled_baseline,
    portion_adjustment_metrics,
    portion_adjustment_score,
)


def _simple_recipe():
    # 3 ingredients, known macro densities per gram
    x0 = np.array([100.0, 50.0, 30.0])
    # protein, fat, carbs, kcal per gram
    M = np.array(
        [
            [0.10, 0.02, 0.01],  # protein
            [0.05, 0.20, 0.01],  # fat
            [0.20, 0.10, 0.70],  # carbs
            [2.0, 3.0, 3.0],     # kcal (simplified)
        ]
    )
    return x0, M


def test_portion_score_zero_when_unchanged():
    r = np.ones(3)
    w = np.ones(3)
    assert portion_adjustment_score(r, w) == pytest.approx(0.0)
    m = portion_adjustment_metrics(r, w)
    assert m["avg_pct_change"] == pytest.approx(0.0)
    assert m["max_pct_change"] == pytest.approx(0.0)


def test_portion_metrics_pct_change():
    r = np.array([1.0, 1.2, 0.8])
    m = portion_adjustment_metrics(r)
    assert m["avg_pct_change"] == pytest.approx(13.333333, rel=0.01)
    assert m["max_pct_change"] == pytest.approx(20.0)


def test_display_metrics_zero_for_uniform_kcal_scaling():
    x0 = np.array([100.0, 50.0, 30.0])
    kcal_before = 600.0
    kcal_target = 400.0
    x_opt = kcal_scaled_baseline(x0, kcal_before, kcal_target)
    m = display_portion_change_metrics(x_opt, x0, kcal_before, kcal_target)
    assert m["portion_score"] == pytest.approx(0.0)
    assert m["avg_pct_change"] == pytest.approx(0.0)
    assert m["max_pct_change"] == pytest.approx(0.0)


def test_display_metrics_nonzero_when_ratios_change():
    x0 = np.array([100.0, 100.0, 100.0])
    kcal_before = 900.0
    kcal_target = 600.0
    x_cal = kcal_scaled_baseline(x0, kcal_before, kcal_target)
    x_opt = x_cal * np.array([1.0, 1.2, 0.8])
    m = display_portion_change_metrics(x_opt, x0, kcal_before, kcal_target)
    assert m["avg_pct_change"] == pytest.approx(13.333333, rel=0.01)
    assert m["max_pct_change"] == pytest.approx(20.0)


def test_serving_display_scales_quantity():
    x0, M = _simple_recipe()
    opt = RecipeMacroOptimizer()
    bounds = MacroBounds(
        protein_g=(10, 20),
        fat_g=(5, 15),
        carbs_g=(20, 40),
        kcal_target=500,
    )
    result = opt.optimize(x0, M, OptimizerConfig(macro_bounds=bounds))
    meta = [
        IngredientMeta(0, "flour", quantity=2.0, unit="cup"),
        IngredientMeta(1, "butter", quantity=0.5, unit="cup"),
        IngredientMeta(2, "sugar", quantity=None, unit="g"),
    ]
    rows = format_serving_display(result, x0, meta)
    assert rows[0]["quantity_optimized"] == pytest.approx(2.0 * result.r[0])
    assert rows[2]["quantity_optimized"] is None


def test_locked_ingredient_unchanged():
    x0, M = _simple_recipe()
    opt = RecipeMacroOptimizer()
    bounds = MacroBounds(
        protein_g=(0, 100),
        fat_g=(0, 100),
        carbs_g=(0, 200),
        kcal_target=500,
    )
    constraints = [
        IngredientConstraint(),
        IngredientConstraint(locked=True),
        IngredientConstraint(),
    ]
    result = opt.optimize(
        x0, M, OptimizerConfig(macro_bounds=bounds), constraints=constraints
    )
    assert result.r[1] == pytest.approx(1.0)
    assert result.x_opt[1] == pytest.approx(x0[1])


def test_optimizer_improves_macro_fit():
    x0, M = _simple_recipe()
    macros_before = compute_macros(x0, M)
    opt = RecipeMacroOptimizer()
    target_protein = macros_before[0] * 1.2
    bounds = MacroBounds(
        protein_g=(target_protein * 0.95, target_protein * 1.05),
        fat_g=(0, 50),
        carbs_g=(0, 100),
        kcal_target=float(macros_before[3]),
    )
    result = opt.optimize(x0, M, OptimizerConfig(macro_bounds=bounds, macro_penalty=50.0))
    assert result.macros_after[0] >= bounds.protein_g[0] * 0.99
    assert result.portion_score >= 0.0


def test_build_macro_matrix_scaling():
    x0 = np.array([200.0])
    per_100g = np.array([[20.0, 10.0, 30.0, 250.0]])
    x, M = build_macro_matrix(x0, per_100g)
    assert compute_macros(x, M)[0] == pytest.approx(40.0)  # 200g * 0.2 protein/g


def test_macro_region_infeasible_returns_fallback():
    x0, M = _simple_recipe()
    bounds = MacroBounds(protein_g=(500, 600), fat_g=(0, 5), carbs_g=(0, 5), kcal_target=200)
    ok, _, _ = check_macro_region_feasible(M, bounds)
    assert ok is False
    result = RecipeMacroOptimizer().optimize(x0, M, OptimizerConfig(macro_bounds=bounds))
    assert result.used_fallback is True
    assert result.macro_feasible is False
    assert result.status == "infeasible_region"
    assert np.allclose(result.r, 1.0)


def test_macro_region_feasible_witness():
    x0, M = _simple_recipe()
    bounds = MacroBounds(
        protein_g=(5, 25),
        fat_g=(5, 20),
        carbs_g=(10, 50),
        kcal_target=400,
    )
    ok, witness, _ = check_macro_region_feasible(M, bounds)
    assert ok is True
    assert witness is not None
    macros = compute_macros(witness, M)
    assert macros[0] >= bounds.protein_g[0]
    assert macros[3] == pytest.approx(bounds.kcal_target, rel=0.05)


def test_already_feasible_skips_optimization():
    x0, M = _simple_recipe()
    macros = compute_macros(x0, M)
    bounds = MacroBounds(
        protein_g=(macros[0] - 1, macros[0] + 1),
        fat_g=(macros[1] - 1, macros[1] + 1),
        carbs_g=(macros[2] - 1, macros[2] + 1),
        kcal_target=float(macros[3]),
    )
    result = RecipeMacroOptimizer().optimize(x0, M, OptimizerConfig(macro_bounds=bounds))
    assert result.already_feasible is True
    assert result.status == "already_feasible"
    assert np.allclose(result.r, 1.0)
    assert result.portion_score == 0.0
