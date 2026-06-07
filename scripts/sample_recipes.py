"""Sample random recipes from the full RecipeNLG CSV (not the fixed 10K slice).

Used by the LLM-judge pilot. Sampling is keyed on the RecipeNLG canonical id
(the unnamed first CSV column) so results join cleanly back to the source rows
regardless of position. A two-pass strategy keeps memory low: pass 1 reads only
the id column to draw a reproducible sample, pass 2 streams the file in chunks to
pull the full rows for the sampled ids.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from ingredient_query_cache import _parse_ingredient_list

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE_CSV = ROOT / "Data" / "recipes" / "RecipeNLG.csv"


def sample_recipe_ids(
    *,
    n: int = 100,
    seed: int = 42,
    recipe_csv: Path = DEFAULT_RECIPE_CSV,
) -> list[int]:
    """Draw `n` random RecipeNLG ids (canonical first-column id), reproducibly."""
    ids = pd.read_csv(recipe_csv, usecols=[0]).iloc[:, 0].astype(int).tolist()
    if n >= len(ids):
        return sorted(ids)
    rng = random.Random(seed)
    return sorted(rng.sample(ids, n))


def load_recipes_by_id(
    ids: list[int],
    *,
    recipe_csv: Path = DEFAULT_RECIPE_CSV,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Load full recipe rows for the given ids; `recipe_id` = canonical NLG id."""
    id_set = set(int(i) for i in ids)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(recipe_csv, chunksize=chunksize):
        id_col = chunk.columns[0]
        sel = chunk[chunk[id_col].astype(int).isin(id_set)]
        if len(sel):
            parts.append(sel)
    if not parts:
        return pd.DataFrame()
    recipes = pd.concat(parts, ignore_index=True)
    id_col = recipes.columns[0]
    recipes = recipes.rename(columns={id_col: "recipe_id"})
    recipes["recipe_id"] = recipes["recipe_id"].astype(int)
    return recipes.sort_values("recipe_id").reset_index(drop=True)


def explode_recipe_ingredients(recipes: pd.DataFrame) -> pd.DataFrame:
    """One row per ingredient line: recipe_id, ingredient_idx, ingredient."""
    df = recipes[["recipe_id", "ingredients"]].copy()
    df["ingredients_list"] = df["ingredients"].map(_parse_ingredient_list)
    out = (
        df[["recipe_id", "ingredients_list"]]
        .explode("ingredients_list")
        .rename(columns={"ingredients_list": "ingredient"})
        .dropna(subset=["ingredient"])
        .reset_index(drop=True)
    )
    out["ingredient"] = out["ingredient"].astype(str)
    out = out[out["ingredient"].str.strip() != ""].reset_index(drop=True)
    out["ingredient_idx"] = out.groupby("recipe_id").cumcount()
    return out


def load_sampled_recipes(
    *,
    n: int = 100,
    seed: int = 42,
    recipe_csv: Path = DEFAULT_RECIPE_CSV,
    ids_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Return (recipes, exploded_ingredients, sampled_ids).

    If `ids_path` exists, reuse the persisted sample for reproducibility;
    otherwise draw a fresh sample and persist it there.
    """
    if ids_path is not None and ids_path.is_file():
        sampled_ids = json.loads(ids_path.read_text())["recipe_ids"]
    else:
        sampled_ids = sample_recipe_ids(n=n, seed=seed, recipe_csv=recipe_csv)
        if ids_path is not None:
            ids_path.parent.mkdir(parents=True, exist_ok=True)
            ids_path.write_text(
                json.dumps({"n": n, "seed": seed, "recipe_ids": sampled_ids}, indent=2) + "\n"
            )

    recipes = load_recipes_by_id(sampled_ids, recipe_csv=recipe_csv)
    recipe_ingredients = explode_recipe_ingredients(recipes)
    return recipes, recipe_ingredients, sampled_ids
