import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mvp_pipeline import UserQuery, run_pipeline


def _mock_corpus():
    return {
        "recipe_ids": [101, 102],
        "recipe_names": ["Soup A", "Salad B"],
        "embeddings": np.array([[1.0] + [0.0] * 383, [0.0, 1.0] + [0.0] * 382], dtype=np.float32),
        "nutrient_rows": [
            {
                "recipe_id": 101,
                "recipe_name": "Soup A",
                "protein_g": 25,
                "total_lipid_fat_g": 15,
                "carbohydrate_by_difference_g": 55,
                "energy_kcal": 500,
            },
            {
                "recipe_id": 102,
                "recipe_name": "Salad B",
                "protein_g": 20,
                "total_lipid_fat_g": 10,
                "carbohydrate_by_difference_g": 40,
                "energy_kcal": 400,
            },
        ],
        "features": {
            101: {"title_clean": "Soup A", "semantic_text": "hearty soup", "ingredient_count": 5},
            102: {"title_clean": "Salad B", "semantic_text": "fresh salad", "ingredient_count": 4},
        },
    }


def _mock_optimize(recipe_id, query, corpus=None):
    return {
        "recipe_id": recipe_id,
        "recipe_name": "Soup A" if recipe_id == 101 else "Salad B",
        "portion_score": 0.01 if recipe_id == 101 else 0.05,
        "avg_pct_change": 2.0 if recipe_id == 101 else 8.0,
        "max_pct_change": 5.0 if recipe_id == 101 else 15.0,
        "optimizer_status": "optimal",
        "sca_iters": 3,
        "converged": True,
        "macros_before": {"protein_g": 20, "fat_g": 10, "carbs_g": 40, "energy_kcal": 400},
        "macros_after": {"protein_g": 22, "fat_g": 11, "carbs_g": 42, "energy_kcal": 420},
        "macro_slack": [0, 0, 0, 0],
        "ingredients": [
            {
                "ingredient": "tomato",
                "fdc_description": "Tomatoes, raw",
                "portion_label": "1 cup",
                "unit": "cup",
                "quantity_original": 2,
                "quantity_optimized": 2.1,
                "adjustment_factor": 1.05,
            }
        ],
        "max_r": 1.05,
    }


@patch("mvp_pipeline.encode_query", return_value=np.ones(384, dtype=np.float32))
@patch("mvp_pipeline.optimize_recipe", side_effect=_mock_optimize)
def test_pipeline_stages_and_mock_judge(mock_opt, mock_enc):
    query = UserQuery(
        taste_text="hearty soup",
        kcal_min=350,
        kcal_max=550,
        fat_frac_min=0.25,
        fat_frac_max=0.35,
        carb_frac_min=0.45,
        carb_frac_max=0.55,
        protein_frac_min=0.15,
        protein_frac_max=0.25,
        top_k=2,
    )
    events = []

    def on_event(ev):
        events.append(ev.stage)

    result = run_pipeline(query, on_event=on_event, log_to_db=False, corpus=_mock_corpus())
    assert "embed_query" in events
    assert "stage1_rank" in events
    assert "optimize" in events
    assert "judge" in events
    assert "done" in events
    assert result["chosen_recipe_id"] == 101
    assert "judge" in result
