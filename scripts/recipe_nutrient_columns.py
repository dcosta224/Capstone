"""Build wide-table column names and DDL for recipe.recipe_nutrients."""

from __future__ import annotations

import re
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    s = str(text or "").strip().lower()
    s = _NON_ALNUM.sub("_", s)
    return s.strip("_") or "nutrient"


def nutrient_column_name(nutrient_id: int, name: str, unit_name: str | None) -> str:
    base = slugify(name)
    unit = slugify(unit_name) if unit_name else "unknown"
    col = f"{base}_{unit}"
    if not col[0].isalpha():
        col = f"n_{col}"
    return col


def assign_nutrient_columns(rows: list[dict[str, Any]]) -> dict[int, str]:
    """Map nutrient_id -> unique SQL column name."""
    seen: dict[str, int] = {}
    out: dict[int, str] = {}
    for row in rows:
        nid = int(row["id"])
        col = nutrient_column_name(nid, str(row["name"]), row.get("unit_name"))
        if col in seen and seen[col] != nid:
            col = f"{col}_n{nid}"
        seen[col] = nid
        out[nid] = col
    return out


def build_recipe_nutrients_ddl(nutrient_col_map: dict[int, str]) -> str:
    fixed = [
        "recipe_id bigint PRIMARY KEY",
        "recipe_name text NOT NULL",
        "n_ingredients integer",
        "total_gram_weight double precision",
        "loaded_at timestamptz NOT NULL DEFAULT now()",
    ]
    nutrient_cols = [
        f'"{col}" double precision'
        for col in sorted(set(nutrient_col_map.values()))
    ]
    cols_sql = ",\n    ".join(fixed + nutrient_cols)
    return f"CREATE TABLE recipe.recipe_nutrients (\n    {cols_sql}\n);"
