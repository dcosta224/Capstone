"""Shared loaders for local diet tagging (USDA CSVs)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foodon_paths import USDA_BRANDED_CSV, USDA_DATA_DIR, USDA_FOOD_CSV

ROOT = Path(__file__).resolve().parents[1]
FOOD_4MACRO_CACHE = ROOT / "scratch" / "food_4macro.csv"
FOOD_NUTRIENT_CSV = USDA_DATA_DIR / "food_nutrient.csv"


def load_foods_catalog(*, limit: int | None = None) -> pd.DataFrame:
    if not USDA_FOOD_CSV.is_file():
        raise FileNotFoundError(f"Missing {USDA_FOOD_CSV}")

    if FOOD_4MACRO_CACHE.is_file():
        catalog = pd.read_csv(FOOD_4MACRO_CACHE, usecols=["fdc_id"])
        fdc_ids = set(catalog["fdc_id"].astype(int).tolist())
        foods = pd.read_csv(
            USDA_FOOD_CSV,
            usecols=["fdc_id", "description"],
            dtype={"description": "string"},
        )
        foods = foods[foods["fdc_id"].isin(fdc_ids)]
        if limit is not None:
            foods = foods.head(limit)
    else:
        foods = pd.read_csv(
            USDA_FOOD_CSV,
            usecols=["fdc_id", "description"],
            dtype={"description": "string"},
            nrows=limit,
        )

    if USDA_BRANDED_CSV.is_file():
        branded = pd.read_csv(
            USDA_BRANDED_CSV,
            usecols=["fdc_id", "ingredients"],
            dtype={"ingredients": "string"},
        )
        foods = foods.merge(branded, on="fdc_id", how="left")
    else:
        foods["ingredients"] = None

    foods["description"] = foods["description"].fillna("").astype(str)
    return foods


def load_nutrients_for_fdc(
    fdc_ids: set[int],
    nutrient_ids: set[int],
) -> dict[tuple[int, int], float]:
    if not FOOD_NUTRIENT_CSV.is_file():
        return {}
    if not fdc_ids or not nutrient_ids:
        return {}

    lookup: dict[tuple[int, int], float] = {}
    chunks = pd.read_csv(
        FOOD_NUTRIENT_CSV,
        usecols=["fdc_id", "nutrient_id", "amount"],
        chunksize=500_000,
        low_memory=False,
    )
    for chunk in chunks:
        sub = chunk[
            chunk["fdc_id"].isin(fdc_ids) & chunk["nutrient_id"].isin(nutrient_ids)
        ]
        for row in sub.itertuples(index=False):
            if pd.notna(row.amount):
                lookup[(int(row.fdc_id), int(row.nutrient_id))] = float(row.amount)
    return lookup


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        df.to_csv(path.with_suffix(".csv"), index=False)
        print(f"parquet unavailable; wrote {path.with_suffix('.csv')}", flush=True)
