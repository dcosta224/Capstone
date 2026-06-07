"""Cache load/save for recipe ingredient matching notebook outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "scratch" / "recipe_matching_10k"

MANIFEST_NAME = "_manifest.json"

INGREDIENT_MATCHES_STAGED = "ingredient_matches_staged.csv"
RECIPE_SUMMARY_STAGED = "recipe_match_summary_staged.csv"
HP_SWEEP_DIRNAME = "hp_sweep"
HP_SWEEP_METRICS = "hp_sweep_metrics.csv"
HP_IDENTITY_LEADERBOARD = "hp_identity_leaderboard.csv"
HP_PREP_LEADERBOARD = "hp_prep_leaderboard.csv"
HP_BEST_CONFIG_JSON = "hp_best_config.json"

# Re-export embedding cache filenames (see ingredient_query_cache.py)
from ingredient_query_cache import (  # noqa: E402
    EMBEDDINGS_META,
    FOOD_DEQUANT_EMB,
    FOOD_DESC_EMBEDDINGS,
    FOOD_NAME_EMB,
    FOOD_PARSED,
    FOOD_PREP_EMB,
    RECIPE_DEQUANT_EMB,
    RECIPE_NAME_EMB,
    RECIPE_PARSED,
    RECIPE_PREP_EMB,
    UNPREPARED_PREP_EMB,
    UNPREPARED_PREP_TEXT,
    load_or_build_unprepared_embedding,
)


def cache_dir(base: Path | None = None) -> Path:
    return base if base is not None else DEFAULT_CACHE_DIR


def manifest_path(dir_path: Path) -> Path:
    return dir_path / MANIFEST_NAME


def read_manifest(dir_path: Path) -> dict[str, Any]:
    path = manifest_path(dir_path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def write_manifest(dir_path: Path, **fields: Any) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    data = read_manifest(dir_path)
    data.update(fields)
    manifest_path(dir_path).write_text(json.dumps(data, indent=2) + "\n")


def _csv_populated(path: Path, *, min_rows: int = 1) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    except OSError:
        return False
    return n >= min_rows


def ingredient_matches_populated(
    path: Path,
    *,
    expected_rows: int | None = None,
    min_rows: int = 1,
) -> bool:
    if not _csv_populated(path, min_rows=min_rows):
        return False
    if expected_rows is None:
        return True
    try:
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    except OSError:
        return False
    return n == expected_rows


def summary_populated(
    path: Path,
    *,
    expected_recipes: int | None = None,
    min_rows: int = 1,
) -> bool:
    if not _csv_populated(path, min_rows=min_rows):
        return False
    if expected_recipes is None:
        return True
    try:
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    except OSError:
        return False
    return n == expected_recipes


def load_ingredient_matches(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_recipe_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_ingredient_matches(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_recipe_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_or_run_ingredient_matches(
    path: Path,
    compute: Callable[[], pd.DataFrame],
    *,
    expected_rows: int | None = None,
    cache_key: str | None = None,
    dir_path: Path | None = None,
) -> pd.DataFrame:
    """Return cached ingredient-match rows or run *compute* and persist."""
    if ingredient_matches_populated(path, expected_rows=expected_rows):
        df = load_ingredient_matches(path)
        print(f"Loaded cache → {path} ({len(df):,} rows)")
        return df

    print(f"Cache miss — running matcher (will save → {path})")
    df = compute()
    save_ingredient_matches(df, path)
    if cache_key and dir_path is not None:
        write_manifest(
            dir_path,
            **{cache_key: {"path": path.name, "n_rows": len(df)}},
        )
    print(f"Saved → {path} ({len(df):,} rows)")
    return df


LEGACY_FILES: dict[str, str] = {
    "ingredient_matches_4macro.csv": INGREDIENT_MATCHES_STAGED,
    "recipe_match_summary_4macro.csv": RECIPE_SUMMARY_STAGED,
}


def migrate_legacy_scratch_csvs(
    dir_path: Path | None = None,
    scratch_root: Path | None = None,
) -> list[str]:
    """Copy old flat scratch/*.csv into cache dir if targets are missing."""
    import shutil

    cache = cache_dir(dir_path)
    root = scratch_root if scratch_root is not None else ROOT / "scratch"
    cache.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for legacy_name, target_name in LEGACY_FILES.items():
        src = root / legacy_name
        dst = cache / target_name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            copied.append(target_name)
    return copied


def load_or_run_summary(
    path: Path,
    compute: Callable[[], pd.DataFrame],
    *,
    expected_recipes: int | None = None,
) -> pd.DataFrame:
    if summary_populated(path, expected_recipes=expected_recipes):
        df = load_recipe_summary(path)
        print(f"Loaded cache → {path} ({len(df):,} recipes)")
        return df

    print(f"Cache miss — building summary (will save → {path})")
    df = compute()
    save_recipe_summary(df, path)
    print(f"Saved → {path} ({len(df):,} recipes)")
    return df
