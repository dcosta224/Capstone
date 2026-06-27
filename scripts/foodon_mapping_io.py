"""Load/save versioned fdc_id -> FoodOn mapping table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from foodon_paths import FOODON_MAPPING_CSV, FOODON_MAPPING_PARQUET

LINKER_VERSION = "v1_tiered_fuzzy_semantic"


def mapping_path(prefer_parquet: bool = True) -> Path:
    if prefer_parquet and FOODON_MAPPING_PARQUET.is_file():
        return FOODON_MAPPING_PARQUET
    return FOODON_MAPPING_CSV


def load_mapping(path: Path | None = None) -> pd.DataFrame:
    p = path or mapping_path()
    if not p.is_file():
        return pd.DataFrame(
            columns=[
                "fdc_id",
                "description",
                "foodon_id",
                "foodon_label",
                "match_method",
                "confidence",
                "fuzzy_score",
                "semantic_score",
                "reviewed",
                "linker_version",
            ]
        )
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load_mapping_lookup(path: Path | None = None) -> dict[int, dict]:
    df = load_mapping(path)
    if df.empty:
        return {}
    out: dict[int, dict] = {}
    for row in df.itertuples(index=False):
        if pd.isna(row.foodon_id) or not str(row.foodon_id).strip():
            continue
        out[int(row.fdc_id)] = {
            "foodon_id": str(row.foodon_id),
            "foodon_label": str(row.foodon_label) if pd.notna(row.foodon_label) else "",
            "match_method": str(row.match_method) if pd.notna(row.match_method) else "",
            "confidence": float(row.confidence) if pd.notna(row.confidence) else 0.0,
        }
    return out


def write_mapping(df: pd.DataFrame) -> Path:
    FOODON_MAPPING_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(FOODON_MAPPING_PARQUET, index=False)
        return FOODON_MAPPING_PARQUET
    except ImportError:
        df.to_csv(FOODON_MAPPING_CSV, index=False)
        return FOODON_MAPPING_CSV


def merge_mapping(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_rows.reset_index(drop=True)
    keep = existing[~existing["fdc_id"].isin(new_rows["fdc_id"])]
    return pd.concat([keep, new_rows], ignore_index=True)
