"""Load usda.food_4macro from Supabase (with local CSV cache)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from db import connect

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = ROOT / "scratch" / "food_4macro.csv"
CACHE_PATH = Path(os.environ.get("FOOD_4MACRO_CACHE", _DEFAULT_CACHE))

FOOD_4MACRO_SQL = """
SELECT
    fdc_id,
    data_type,
    description,
    food_category_id,
    publication_date
FROM usda.food_4macro
ORDER BY fdc_id
"""


def load_food_4macro(*, refresh: bool = False) -> pd.DataFrame:
    """Return rows from materialized view usda.food_4macro."""
    if CACHE_PATH.exists() and not refresh:
        out = pd.read_csv(
            CACHE_PATH,
            dtype={"fdc_id": "int64", "data_type": "string"},
        )
        out["description"] = out["description"].fillna("").astype(str)
        return out

    with connect() as conn:
        out = pd.read_sql(FOOD_4MACRO_SQL, conn)

    out["description"] = out["description"].fillna("").astype(str)
    out = out.sort_values("fdc_id").reset_index(drop=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE_PATH, index=False)
    return out
