"""Load non-branded USDA food catalog from local CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_food_mvp_density_inferable import BRANDED_TYPE, DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
FOOD_CACHE = ROOT / "scratch" / "food_non_branded.csv"
MATCHING_CACHE = ROOT / "scratch" / "food_non_branded_matching.csv"


def load_food_non_branded(*, refresh: bool = False) -> pd.DataFrame:
    """All non-branded rows from food.csv."""
    if FOOD_CACHE.exists() and not refresh:
        out = pd.read_csv(FOOD_CACHE, dtype={"fdc_id": "int64", "data_type": "string"})
        out["description"] = out["description"].fillna("").astype(str)
        return out

    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id", "data_type", "description", "food_category_id", "publication_date"],
        dtype={"fdc_id": "int64", "data_type": "string"},
        low_memory=False,
    )
    out = food.loc[food["data_type"] != BRANDED_TYPE].copy()
    out["description"] = out["description"].fillna("").astype(str)
    out = out.sort_values("fdc_id").reset_index(drop=True)
    FOOD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FOOD_CACHE, index=False)
    return out


def load_food_non_branded_matching(*, refresh: bool = False) -> pd.DataFrame:
    """Non-branded foods shaped for FoodMatcher (description as match name).

    USDA descriptions are already normalized titles; parsing 100k+ lines with
    ingredient-parser is slow and often mis-parses them. Recipe ingredients
    are still parsed the same way as in the food_mvp run.
    """
    if MATCHING_CACHE.exists() and not refresh:
        return pd.read_csv(MATCHING_CACHE, dtype={"fdc_id": "int64", "data_type": "string"})

    from food_name_prefixes import words_before_delimiters

    raw = load_food_non_branded()
    out = raw.copy()
    out["ingredient"] = out["description"]
    out["name"] = out["description"]
    out["size"] = None
    out["preparation"] = None
    out["prefix"] = out["description"].map(words_before_delimiters)
    out["parse_status"] = "catalog_description"
    out["parse_method"] = None
    out.to_csv(MATCHING_CACHE, index=False)
    return out
