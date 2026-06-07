"""Load food_mvp (density-inferable) rows from local USDA CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_food_mvp_density_inferable import (
    BRANDED_TYPE,
    DATA_DIR,
    collect_fdc_ids,
    density_inferable_fdc_ids,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "scratch" / "food_mvp_inferable.csv"


def load_food_mvp_inferable(*, refresh: bool = False) -> pd.DataFrame:
    """Return food_mvp rows: non-branded, nutrients, portions, density-inferable."""
    if CACHE_PATH.exists() and not refresh:
        return pd.read_csv(CACHE_PATH, dtype={"fdc_id": "int64", "data_type": "string"})

    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id", "data_type", "description", "food_category_id", "publication_date"],
        dtype={"fdc_id": "int64", "data_type": "string"},
        low_memory=False,
    )
    universe = set(food.loc[food["data_type"] != BRANDED_TYPE, "fdc_id"])
    nutrient_ids = collect_fdc_ids(DATA_DIR / "food_nutrient.csv", universe)
    portion_ids = collect_fdc_ids(DATA_DIR / "food_portion.csv", universe)
    mvp_ids = nutrient_ids & portion_ids
    inferable_ids = density_inferable_fdc_ids(mvp_ids)

    out = food[food["fdc_id"].isin(inferable_ids)].copy()
    out["description"] = out["description"].fillna("")
    out = out.sort_values("fdc_id").reset_index(drop=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE_PATH, index=False)
    return out
