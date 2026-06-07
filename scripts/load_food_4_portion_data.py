"""Load usda.food_4_portion_data from Supabase (with local CSV cache)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from db import connect

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "scratch" / "food_4_portion_data.csv"

FOOD_4_PORTION_SQL = """
SELECT
    fdc_id,
    data_type,
    description,
    food_category_id,
    publication_date
FROM usda.food_4_portion_data
ORDER BY fdc_id
"""


def load_food_4_portion_data(*, refresh: bool = False) -> pd.DataFrame:
    """Return rows from materialized view usda.food_4_portion_data."""
    if CACHE_PATH.exists() and not refresh:
        out = pd.read_csv(
            CACHE_PATH,
            dtype={"fdc_id": "int64", "data_type": "string"},
        )
        out["description"] = out["description"].fillna("").astype(str)
        return out

    with connect() as conn:
        out = pd.read_sql(FOOD_4_PORTION_SQL, conn)

    out["description"] = out["description"].fillna("").astype(str)
    out = out.sort_values("fdc_id").reset_index(drop=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE_PATH, index=False)
    return out
