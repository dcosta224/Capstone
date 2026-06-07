"""Recipe text cleaning and semantic_text construction (from exploration.ipynb)."""

from __future__ import annotations

import ast
import json
import re
from typing import Iterable

import pandas as pd

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384

UNITS = (
    r"\b(c|cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|"
    r"oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|ml|l|"
    r"pkg|package|packages|can|cans|jar|jars)\b"
)


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def clean_ingredient(item: object) -> str:
    if pd.isna(item):
        return ""

    item = str(item).lower()
    item = re.sub(r"\([^)]*\)", " ", item)
    item = re.sub(r"\d+\s*/\s*\d+|\d+\.\d+|\d+", " ", item)
    item = re.sub(UNITS, " ", item)
    item = re.sub(r"[^a-z\s]", " ", item)
    return " ".join(item.split())


def parse_json_list(raw: object) -> list:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)

    # psycopg2 / pandas may return numpy arrays for JSON list columns
    try:
        import numpy as np

        if isinstance(raw, np.ndarray):
            return raw.tolist()
    except ImportError:
        pass

    try:
        if pd.isna(raw):
            return []
    except (ValueError, TypeError):
        pass

    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")
    return parsed


def canonical_json_field(raw: object) -> str:
    """Stable string for dedup keys (DB may return lists; CSV returns JSON text)."""
    return json.dumps(parse_json_list(raw), ensure_ascii=False, separators=(",", ":"))


def parse_ingredients(raw: object) -> list[str]:
    return [str(x) for x in parse_json_list(raw)]


def clean_ingredients(raw: object) -> list[str]:
    return [clean_ingredient(x) for x in parse_ingredients(raw)]


def build_semantic_text(title_clean: str, ingredients_clean: Iterable[str]) -> str:
    tokens = " ".join(sorted(ingredients_clean))
    return f"{title_clean} | {tokens}".strip()


def prepare_recipe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return deduplicated rows with id, title_clean, semantic_text, ingredient_count."""
    required = {"id", "title", "ingredients", "directions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    work = df.loc[:, ["id", "title", "ingredients", "directions"]].copy()
    work["id"] = work["id"].astype("int64")
    work["title"] = work["title"].astype(str)
    work["ingredients"] = work["ingredients"].map(canonical_json_field)
    work["directions"] = work["directions"].map(canonical_json_field)
    work = work.drop_duplicates(subset=["title", "ingredients", "directions"], keep="first")

    work["title_clean"] = work["title"].map(normalize_text)
    work["ingredients_clean"] = work["ingredients"].map(clean_ingredients)
    work["semantic_text"] = work.apply(
        lambda row: build_semantic_text(row["title_clean"], row["ingredients_clean"]),
        axis=1,
    )
    work["ingredient_count"] = work["ingredients_clean"].map(len)

    return work.loc[
        :,
        ["id", "title_clean", "semantic_text", "ingredient_count"],
    ].reset_index(drop=True)


def vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"
