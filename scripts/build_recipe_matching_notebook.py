#!/usr/bin/env python3
"""Regenerate scratch/food_mvp_recipe_matching.ipynb for staged hybrid matching."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "food_mvp_recipe_matching.ipynb"


def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def cell_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        cell_md(
            """# Staged ingredient matching (10K RecipeNLG × `food_4macro`)

Iterate on a **two-stage hybrid** matcher:
1. **Base identity** — mostly lexical (+ small semantic) on parsed **name**
2. **Prep / version** — semantic + lexical prep, modifier rules, default/basic bonus

**One-time upstream steps** (cached under `scratch/recipe_matching_10k/`):
- Parse recipe lines and `food_4macro` descriptions with `ingredient-parser-nlp`
- Embed **name**, **preparation**, and **dequantified** text separately (`all-MiniLM-L6-v2`)
- Empty recipe preparation → shared **unprepared** embedding (one vector, embedded once)

Implementation: [`scripts/ingredient_match_staged.py`](../scripts/ingredient_match_staged.py), [`scripts/ingredient_query_cache.py`](../scripts/ingredient_query_cache.py)
"""
        ),
        cell_code(
            '''import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from ingredient_match_staged import (
    StagedMatchConfig,
    StagedFoodIndex,
    match_ingredients_staged,
)
from ingredient_query_cache import (
    ensure_hf_token,
    load_or_build_food_artifacts,
    load_or_build_recipe_artifacts,
)
from progress_utils import force_std_tqdm

force_std_tqdm()  # text tqdm in Jupyter (no ipywidgets bars)
from load_food_4macro import load_food_4macro
from recipe_match_cache import (
    DEFAULT_CACHE_DIR,
    INGREDIENT_MATCHES_STAGED,
    RECIPE_SUMMARY_STAGED,
    load_or_run_ingredient_matches,
    load_or_run_summary,
)
import recipe_match_summary as recipe_match_metrics

summarize_recipe_matches = recipe_match_metrics.summarize_recipe_matches

WORK_DIR = DEFAULT_CACHE_DIR
WORK_DIR.mkdir(parents=True, exist_ok=True)

ensure_hf_token()  # HF_TOKEN from repo .env

# Tunable weights — edit and re-run §3–5 without rebuilding embeddings
MATCH_CONFIG = StagedMatchConfig()
'''
        ),
        cell_md("## 1. Load recipes and explode ingredient lines"),
        cell_code(
            '''RECIPE_NLG_PATH = ROOT / "Data/recipes/RecipeNLG.csv"
RECIPE_NROWS = 10_000


def parse_ingredient_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    text = text.strip()
    if not text:
        return []
    if '", "' not in text:
        item = text.strip('"')
        return [item] if item else []
    parts = text.split('", "')
    out: list[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if i == 0:
            part = part.removeprefix('["').removeprefix('"')
        if i == len(parts) - 1:
            part = part.removesuffix('"]').removesuffix('"')
        if part:
            out.append(part)
    return out


recipes = pd.read_csv(RECIPE_NLG_PATH, nrows=RECIPE_NROWS)
recipes["ingredients_list"] = recipes["ingredients"].map(parse_ingredient_list)
recipes["recipe_id"] = recipes.index

recipe_ingredients = (
    recipes[["recipe_id", "ingredients_list"]]
    .explode("ingredients_list")
    .rename(columns={"ingredients_list": "ingredient"})
    .reset_index(drop=True)
)
recipe_ingredients["ingredient_idx"] = recipe_ingredients.groupby("recipe_id").cumcount()

print(f"recipes: {len(recipes):,}")
print(f"ingredient lines: {len(recipe_ingredients):,}")
recipe_ingredients.head(3)
'''
        ),
        cell_md(
            """## 2. One-shot parse + triple embeddings (cached)

Three vectors per recipe line and per `food_4macro` row: **name**, **preparation**, **dequantified** (original text with quantity numerals stripped).

Skipped when parquet + all six `.npy` files exist with matching row counts.

Long steps show **tqdm** progress bars (parsing, embedding, index build).
"""
        ),
        cell_code(
            '''parsed_ingredients, name_embeddings, prep_embeddings, dequant_embeddings, recipe_meta = (
    load_or_build_recipe_artifacts(recipe_ingredients, WORK_DIR)
)

food_4macro_raw = load_food_4macro()
print(f"food_4macro rows: {len(food_4macro_raw):,}")

food_parsed, food_name_emb, food_prep_emb, food_dequant_emb, food_meta = load_or_build_food_artifacts(
    food_4macro_raw,
    WORK_DIR,
)

food_index = StagedFoodIndex.from_catalog(
    food_4macro_raw,
    name_embeddings=food_name_emb,
    prep_embeddings=food_prep_emb,
    dequant_embeddings=food_dequant_emb,
    config=MATCH_CONFIG,
    show_progress=True,
)

print("recipe embeddings:", recipe_meta)
if "prep_used_unprepared" in parsed_ingredients.columns:
    n_unprep = int(parsed_ingredients["prep_used_unprepared"].sum())
    print(f"recipe prep: {n_unprep:,} lines use unprepared proxy (empty preparation)")
print("food embeddings:", food_meta)
print(f"food index: {len(food_index.candidates):,} candidates")
parsed_ingredients[
  ["recipe_id", "ingredient", "name", "preparation", "prep_used_unprepared", "dequantified"]
].head(8)
'''
        ),
        cell_md(
            """## 3. Staged hybrid matching

**Stage 1 — base identity** (0.80 lexical / 0.20 semantic on name)  
**Stage 2 — prep/version** (semantic + lexical prep + modifier rules + default bonus)  
**Final** = 0.65×base + 0.25×prep + 0.10×default − unsupported-modifier penalties
"""
        ),
        cell_code(
            '''MATCHES_PATH = WORK_DIR / INGREDIENT_MATCHES_STAGED
_n_ing = len(parsed_ingredients)


def _run_staged_match():
    return match_ingredients_staged(
        parsed_ingredients,
        name_embeddings,
        prep_embeddings,
        dequant_embeddings,  # required — dequantified ingredient text
        food_index,  # required — built in §2 from food_4macro embeddings
        show_progress=True,
    )


ingredient_matches = load_or_run_ingredient_matches(
    MATCHES_PATH,
    _run_staged_match,
    expected_rows=_n_ing,
    cache_key="ingredient_matches_staged",
    dir_path=WORK_DIR,
)

ingredient_matches[
    [
        "recipe_id",
        "name",
        "preparation",
        "matched_description",
        "match_score",
        "base_score",
        "prep_score",
        "modifier_penalty",
        "fallback_reason",
        "match_quality",
    ]
].head(12)
'''
        ),
        cell_md("## 4. Aggregate quality"),
        cell_code(
            '''total = len(ingredient_matches)
resolved = ingredient_matches["matched_fdc_id"].notna().sum()
summary = pd.Series(
    {
        "match_rate_pct": round(100 * resolved / total, 2),
        "high": (ingredient_matches["match_quality"] == "high").sum(),
        "medium": (ingredient_matches["match_quality"] == "medium").sum(),
        "low": (ingredient_matches["match_quality"] == "low").sum(),
        "unresolved": (ingredient_matches["match_quality"] == "unresolved").sum(),
        "avg_match_score": round(ingredient_matches["match_score"].mean(), 4),
        "avg_base_score": round(ingredient_matches["base_score"].mean(), 4),
        "avg_prep_score": round(ingredient_matches["prep_score"].mean(), 4),
    }
)
display(summary.to_frame("value"))
display(ingredient_matches["match_quality"].value_counts().to_frame("count"))
display(ingredient_matches["fallback_reason"].value_counts(dropna=False).head(10).to_frame("count"))
'''
        ),
        cell_md("## 5. Per-recipe summary"),
        cell_code(
            '''SUMMARY_PATH = WORK_DIR / RECIPE_SUMMARY_STAGED

recipe_match_summary = load_or_run_summary(
    SUMMARY_PATH,
    lambda: summarize_recipe_matches(ingredient_matches, recipes),
    expected_recipes=RECIPE_NROWS,
)

with_ing = recipe_match_summary.loc[recipe_match_summary["n_ingredients"] > 0]
display(
    with_ing[
        [
            "percent_high",
            "percent_medium",
            "percent_low",
            "percent_unmatched",
            "percent_med_high",
            "avg_match_score",
        ]
    ].mean().round(2)
)
recipe_match_summary.head(8)
'''
        ),
        cell_md(
            """## 6. Staged hyperparameter grid search

**Two-phase grid** (evaluation is per-stage, not one blended score):

| Phase | Grid | Rank by | Metrics prefix |
|-------|------|---------|----------------|
| 1 Identity | `name_sem` × `dequant_sem` | `stage1_avg` (`base_score`) | `stage1_*`, `name_channel_*`, `dequant_channel_*` |
| 2 Prep | `prep_sem` (best identity fixed) | `stage2_avg` (`prep_score`) | `stage2_*` |

`final_*` columns are reported for reference only — **not** used to pick HP.

Outputs under `hp_sweep/`: leaderboards, per-config match CSVs, `hp_best_config.json`.
"""
        ),
        cell_code(
            '''from ingredient_match_hp_sweep import (
    QUICK_SEMANTIC_GRID,
    STAGE1_RANK_KEY,
    STAGE2_RANK_KEY,
    hp_sweep_dir,
    run_staged_hp_grid_search,
)
from recipe_match_cache import (
    HP_BEST_CONFIG_JSON,
    HP_IDENTITY_LEADERBOARD,
    HP_PREP_LEADERBOARD,
)

HP_GRID = QUICK_SEMANTIC_GRID  # 3×3 identity + 3 prep = 12 runs; DEFAULT_SEMANTIC_GRID → 25 + 5 = 30

hp_result = run_staged_hp_grid_search(
    parsed_ingredients,
    name_embeddings,
    prep_embeddings,
    dequant_embeddings,
    food_index,
    work_dir=WORK_DIR,
    identity_grid=HP_GRID,
    prep_grid=HP_GRID,
    base_config=MATCH_CONFIG,
    force=False,
    show_progress=True,
)

hp_dir = hp_result["hp_dir"]
identity_lb = hp_result["identity_leaderboard"]
prep_lb = hp_result["prep_leaderboard"]
best_config = hp_result["best_config"]
print(f"HP dir: {hp_dir}")
print(f"Best config slug: {hp_result['best_payload']['config_slug']}")
'''
        ),
        cell_code(
            '''# Phase 1 leaderboard — identity / base stage only
identity_cols = [
    "config_slug",
    "base_name_semantic_weight",
    "base_dequant_semantic_weight",
    STAGE1_RANK_KEY,
    "stage1_pct_gte_0_55",
    "stage1_quality_high_pct",
    "name_channel_avg",
    "dequant_channel_avg",
    "matches_file",
]
display(identity_lb[identity_cols].head(12))
'''
        ),
        cell_code(
            '''# Phase 2 leaderboard — prep stage (identity weights fixed to phase-1 winner)
prep_cols = [
    "config_slug",
    "prep_semantic_weight",
    "prep_lexical_weight",
    STAGE2_RANK_KEY,
    "stage2_pct_gte_0_55",
    "stage2_quality_high_pct",
    "final_avg",
    "matches_file",
]
display(prep_lb[prep_cols].head(12))
'''
        ),
        cell_code(
            '''# Apply best HP to food index for §3-style matching (optional)
food_index.config = best_config
MATCH_CONFIG = best_config

best_matches_path = hp_dir / hp_result["best_payload"]["best_matches_file"]
print(f"Best match CSV for error analysis: {best_matches_path}")
print(f"Config JSON: {hp_dir / HP_BEST_CONFIG_JSON}")
'''
        ),
        cell_md("## 7. Error analysis samples"),
        cell_code(
            '''# Fallback / low-confidence rows
fallback = ingredient_matches[
    ingredient_matches["fallback_reason"].notna()
    | (ingredient_matches["match_quality"].isin(["low", "unresolved"]))
].copy()

cols = [
    "recipe_id",
    "ingredient",
    "name",
    "preparation",
    "matched_description",
    "match_score",
    "base_score",
    "prep_score",
    "modifier_penalty",
    "fallback_reason",
    "match_quality",
]
display(fallback[cols].sample(20, random_state=0) if len(fallback) else fallback)

# High modifier penalty (likely over-specific food vs simple query)
penalized = ingredient_matches.nlargest(15, "modifier_penalty")
display(penalized[cols])
'''
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
