"""Tests for neighborhood-mean macro target suggestions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recipe_opt_agent.macro_target_suggestions import (
    _rounded_box_pm,
    suggest_macro_targets,
)


def _synthetic_lines() -> pd.DataFrame:
    rows = []
    for rid, items in {
        "rep": [(1, 120.0), (2, 30.0)],
        "other": [(1, 80.0), (3, 50.0)],
    }.items():
        for idx, (fdc, g) in enumerate(items):
            rows.append(
                {
                    "recipe_nlg_id": rid,
                    "ingredient_idx": idx,
                    "fdc_id": fdc,
                    "gram_weight": g,
                    "fdc_description": f"food_{fdc}",
                }
            )
    return pd.DataFrame(rows)


def test_suggest_neighborhood_mean_only(monkeypatch):
    per_g = {
        1: np.array([0.10, 0.02, 0.75, 3.6]),
        2: np.array([0.05, 0.40, 0.10, 4.1]),
        3: np.array([0.40, 0.05, 0.05, 2.2]),
    }
    monkeypatch.setattr(
        "recipe_opt_agent.problem_loader._per_gram_macros_for_fdc_ids",
        lambda _ids: per_g,
    )

    result = suggest_macro_targets(_synthetic_lines(), "rep")
    assert result["n_recipes"] == 2
    presets = result["presets"]
    assert list(presets.keys()) == ["neighborhood_mean"]

    mean = presets["neighborhood_mean"]
    assert mean["pad_pct"] == 5
    for key in ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max"):
        assert abs(mean["box"][key] * 100 - round(mean["box"][key] * 100)) < 1e-9
    assert round((mean["box"]["protein_max"] - mean["box"]["protein_min"]) * 100) == 10


def test_high_protein_targets_from_mean():
    from recipe_opt_agent.macro_target_suggestions import high_protein_targets_from_mean

    hp = high_protein_targets_from_mean((0.18, 0.47, 0.35), pad_pct=2)
    mid = hp["midpoint"]
    # +10pp protein, −5pp carbs/fat → ~0.28 / 0.42 / 0.30
    assert abs(mid["protein"] - 0.28) <= 0.02
    assert abs(mid["carbs"] - 0.42) <= 0.02
    assert abs(mid["fat"] - 0.30) <= 0.02
    box = hp["box"]
    assert box["protein_max"] - box["protein_min"] == pytest.approx(0.04)
    assert box["protein_min"] < box["protein_max"]
    # Feasible simplex intersection
    assert box["protein_max"] + box["carb_max"] + box["fat_max"] >= 1.0 - 1e-9
    assert box["protein_min"] + box["carb_min"] + box["fat_min"] <= 1.0 + 1e-9


def test_rounded_box_pm():
    box = _rounded_box_pm((0.183, 0.476, 0.341), pad_pct=5)
    # 18/48/34 ± 5
    assert box["protein_min"] == 0.13
    assert box["protein_max"] == 0.23
    assert box["carb_min"] == 0.43
    assert box["carb_max"] == 0.53
    assert box["fat_min"] == 0.29
    assert box["fat_max"] == 0.39


def test_validate_macro_box():
    from recipe_opt_web.server import validate_macro_box

    # Valid default box.
    assert validate_macro_box(0.19, 0.23, 0.345, 0.545, 0.245, 0.445) == []
    # Maxes sum to 90% < 100% → infeasible.
    errs = validate_macro_box(0.10, 0.30, 0.10, 0.30, 0.10, 0.30)
    assert any("sum to 90%" in e for e in errs)
    # Mins sum to 120% > 100% → infeasible.
    errs = validate_macro_box(0.40, 0.60, 0.40, 0.60, 0.40, 0.60)
    assert any("120%" in e for e in errs)
    # min > max on one axis.
    errs = validate_macro_box(0.25, 0.20, 0.345, 0.545, 0.245, 0.445)
    assert any("protein min" in e for e in errs)
