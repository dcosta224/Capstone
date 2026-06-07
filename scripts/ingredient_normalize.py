"""Apply shared ingredient parsing schema to a DataFrame column."""

from __future__ import annotations

import pandas as pd

from parse_recipe_ingredient import PARSE_FIELDS, parse_ingredient_fields


def apply_ingredient_parse(df: pd.DataFrame, text_col: str = "ingredient") -> pd.DataFrame:
    """Parse `text_col` and append quantity/unit/name/... columns."""
    parsed = df[text_col].map(parse_ingredient_fields)
    parsed_df = pd.DataFrame(parsed.tolist())
    return pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
