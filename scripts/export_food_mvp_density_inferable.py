#!/usr/bin/env python3
"""Export food_mvp foods with density-inferable portions (matches usda_eda.ipynb logic)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from usda_volume_units import text_has_volume

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "All_Food_Data_April_2026"
OUT_PATH = ROOT / "scratch" / "food_mvp_volume_gram_weight.txt"
BRANDED_TYPE = "branded_food"
CHUNK_SIZE = 100_000

MEASURE_UNITS = pd.read_csv(DATA_DIR / "measure_unit.csv", dtype={"id": "string", "name": "string"})
MU_BY_ID = dict(zip(MEASURE_UNITS["id"], MEASURE_UNITS["name"]))


def collect_fdc_ids(path: Path, universe: set[int]) -> set[int]:
    seen: set[int] = set()
    for chunk in pd.read_csv(path, usecols=["fdc_id"], dtype={"fdc_id": "string"}, chunksize=500_000):
        ids = pd.to_numeric(chunk["fdc_id"], errors="coerce").dropna().astype("int64")
        seen.update(ids[ids.isin(universe)].unique())
    return seen


def density_inferable_fdc_ids(mvp_ids: set[int]) -> set[int]:
    """gram_weight > 0 and volume token in modifier, portion_description, or measure_unit."""
    inferable: set[int] = set()
    usecols = ["fdc_id", "modifier", "measure_unit_id", "portion_description", "gram_weight"]
    for chunk in pd.read_csv(
        DATA_DIR / "food_portion.csv",
        usecols=usecols,
        dtype="string",
        chunksize=CHUNK_SIZE,
    ):
        chunk["fdc_id"] = pd.to_numeric(chunk["fdc_id"], errors="coerce")
        chunk = chunk[chunk["fdc_id"].isin(mvp_ids)]
        if chunk.empty:
            continue
        modifier = chunk["modifier"].fillna("").str.strip()
        description = chunk["portion_description"].fillna("").str.strip()
        measure_unit = chunk["measure_unit_id"].map(MU_BY_ID).fillna("")
        gram_weight = pd.to_numeric(chunk["gram_weight"], errors="coerce")
        has_gram_weight = gram_weight.notna() & (gram_weight > 0)
        has_volume = [
            text_has_volume(m, d, u)
            for m, d, u in zip(modifier, description, measure_unit, strict=True)
        ]
        ok = has_gram_weight & pd.Series(has_volume, index=chunk.index)
        inferable.update(chunk.loc[ok, "fdc_id"].astype(int).tolist())
    return inferable


def cup_modifier_fdc_ids(mvp_ids: set[int]) -> set[int]:
    """Narrow filter: modifier is exactly 'cup' or 'cups' (case-insensitive) + gram_weight > 0."""
    cup_ids: set[int] = set()
    for chunk in pd.read_csv(
        DATA_DIR / "food_portion.csv",
        usecols=["fdc_id", "modifier", "gram_weight"],
        dtype="string",
        chunksize=CHUNK_SIZE,
    ):
        chunk["fdc_id"] = pd.to_numeric(chunk["fdc_id"], errors="coerce")
        chunk = chunk[chunk["fdc_id"].isin(mvp_ids)]
        gw = pd.to_numeric(chunk["gram_weight"], errors="coerce")
        is_cup = chunk["modifier"].fillna("").str.strip().str.lower().isin(["cup", "cups"])
        ok = is_cup & gw.notna() & (gw > 0)
        cup_ids.update(chunk.loc[ok, "fdc_id"].astype(int).tolist())
    return cup_ids


def main() -> None:
    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id", "data_type", "description"],
        dtype={"fdc_id": "int64", "data_type": "string"},
    )
    universe = set(food.loc[food["data_type"] != BRANDED_TYPE, "fdc_id"])
    nutrient_ids = collect_fdc_ids(DATA_DIR / "food_nutrient.csv", universe)
    portion_ids = collect_fdc_ids(DATA_DIR / "food_portion.csv", universe)
    mvp_ids = nutrient_ids & portion_ids

    inferable_ids = density_inferable_fdc_ids(mvp_ids)
    cup_ids = cup_modifier_fdc_ids(mvp_ids)

    rows = food[food["fdc_id"].isin(inferable_ids)].copy()
    names = (
        rows["description"]
        .fillna("")
        .sort_values(kind="stable")
        .tolist()
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Density-inferable foods: {len(inferable_ids):,}")
    print(f"Cup/cups modifier only:  {len(cup_ids):,}")
    print(f"Wrote {len(names):,} names → {OUT_PATH}")


if __name__ == "__main__":
    main()
