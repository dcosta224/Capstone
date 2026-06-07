"""Staged grid search over lexical vs semantic weights with per-stage evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ingredient_match_staged import (
    StagedFoodIndex,
    StagedMatchConfig,
    config_slug,
    match_ingredients_staged,
)
from ingredient_query_cache import (
    DEFAULT_WORK_DIR,
    load_or_build_food_artifacts,
    load_or_build_recipe_artifacts,
    load_recipe_ingredients,
)
from load_food_4macro import load_food_4macro
from progress_utils import _tqdm_factory, _tqdm_kwargs
from recipe_match_cache import save_ingredient_matches
from recipe_match_summary import summarize_staged_match_metrics

ROOT = Path(__file__).resolve().parents[1]
HP_SWEEP_DIRNAME = "hp_sweep"
HP_IDENTITY_LEADERBOARD = "hp_identity_leaderboard.csv"
HP_PREP_LEADERBOARD = "hp_prep_leaderboard.csv"
HP_BEST_CONFIG_JSON = "hp_best_config.json"
HP_MATCHES_PREFIX = "ingredient_matches_"

DEFAULT_SEMANTIC_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
QUICK_SEMANTIC_GRID = (0.0, 0.5, 1.0)

# Staged HP ranking keys (not the blended final score)
STAGE1_RANK_KEY = "stage1_avg"
STAGE2_RANK_KEY = "stage2_avg"


def hp_sweep_dir(work_dir: Path | None = None) -> Path:
    return Path(work_dir or DEFAULT_WORK_DIR) / HP_SWEEP_DIRNAME


def identity_slug(name_sem: float, dequant_sem: float) -> str:
    return f"nS{name_sem:.2f}_dS{dequant_sem:.2f}"


def matches_path_for_config(hp_dir: Path, config: StagedMatchConfig, *, phase: str = "") -> Path:
    slug = config_slug(config)
    if phase:
        return hp_dir / f"{HP_MATCHES_PREFIX}{phase}_{slug}.csv"
    return hp_dir / f"{HP_MATCHES_PREFIX}{slug}.csv"


def iter_identity_configs(
    semantic_grid: tuple[float, ...],
    *,
    base_config: StagedMatchConfig | None = None,
    prep_semantic: float | None = None,
) -> Iterator[StagedMatchConfig]:
    """Grid: name_sem × dequant_sem (prep weight held fixed)."""
    base = base_config or StagedMatchConfig()
    prep_sem = 0.5 if prep_semantic is None else prep_semantic
    for name_sem, dequant_sem in itertools.product(semantic_grid, semantic_grid):
        yield base.with_lexical_semantic_grid(
            name_semantic=name_sem,
            dequant_semantic=dequant_sem,
            prep_semantic=prep_sem,
        )


def iter_prep_configs(
    semantic_grid: tuple[float, ...],
    identity_config: StagedMatchConfig,
) -> Iterator[StagedMatchConfig]:
    """Grid: prep_sem with identity (name/dequant) weights fixed from phase 1."""
    for prep_sem in semantic_grid:
        yield identity_config.with_lexical_semantic_grid(
            name_semantic=identity_config.base_name_semantic_weight,
            dequant_semantic=identity_config.base_dequant_semantic_weight,
            prep_semantic=prep_sem,
        )


def _run_one_config(
    config: StagedMatchConfig,
    *,
    parsed_ingredients: pd.DataFrame,
    name_embeddings: np.ndarray,
    prep_embeddings: np.ndarray,
    dequant_embeddings: np.ndarray,
    food_index: StagedFoodIndex,
    out_path: Path,
    force: bool,
    show_progress: bool,
    progress_desc: str,
) -> pd.DataFrame:
    if not force and out_path.is_file() and out_path.stat().st_size > 0:
        return pd.read_csv(out_path, low_memory=False)

    food_index.config = config
    matches = match_ingredients_staged(
        parsed_ingredients,
        name_embeddings,
        prep_embeddings,
        dequant_embeddings,
        food_index,
        show_progress=show_progress,
        progress_desc=progress_desc,
        progress_position=1 if show_progress else None,
        progress_leave=False,
    )
    slug = config_slug(config)
    matches["hp_config_slug"] = slug
    for col, val in config.hp_fields().items():
        matches[col] = val
    save_ingredient_matches(matches, out_path)
    return matches


def _metrics_row(
    config: StagedMatchConfig,
    metrics: dict[str, Any],
    *,
    phase: str,
    run_index: int,
    total_runs: int,
    matches_path: Path,
    rank_key: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "run_index": run_index,
        "total_runs": total_runs,
        "config_slug": config_slug(config),
        "rank_key": rank_key,
        "matches_file": matches_path.name,
        **config.hp_fields(),
        **metrics,
    }
    return row


def _print_staged_row(row: dict[str, Any], *, phase: str) -> None:
    if phase == "identity":
        parts = [
            f"[{row.get('run_index')}/{row.get('total_runs')}]",
            row.get("config_slug"),
            f"stage1_avg={row.get('stage1_avg')}",
            f"stage1_pct_gte_0_55={row.get('stage1_pct_gte_0_55')}",
            f"name_ch_avg={row.get('name_channel_avg')}",
            f"dequant_ch_avg={row.get('dequant_channel_avg')}",
        ]
    else:
        parts = [
            f"[{row.get('run_index')}/{row.get('total_runs')}]",
            row.get("config_slug"),
            f"stage2_avg={row.get('stage2_avg')}",
            f"stage2_pct_gte_0_55={row.get('stage2_pct_gte_0_55')}",
            f"final_avg={row.get('final_avg')}",
        ]
    print(" | ".join(str(p) for p in parts), flush=True)


def _config_from_identity_row(row: pd.Series, base: StagedMatchConfig) -> StagedMatchConfig:
    return base.with_lexical_semantic_grid(
        name_semantic=float(row["base_name_semantic_weight"]),
        dequant_semantic=float(row["base_dequant_semantic_weight"]),
        prep_semantic=0.5,
    )


def run_staged_hp_grid_search(
    parsed_ingredients: pd.DataFrame,
    name_embeddings: np.ndarray,
    prep_embeddings: np.ndarray,
    dequant_embeddings: np.ndarray,
    food_index: StagedFoodIndex,
    *,
    work_dir: Path | None = None,
    identity_grid: tuple[float, ...] = DEFAULT_SEMANTIC_GRID,
    prep_grid: tuple[float, ...] | None = None,
    base_config: StagedMatchConfig | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
  Two-phase staged grid search:

  1. **Identity** — grid name_sem × dequant_sem; rank by ``stage1_*`` (base_score only).
  2. **Prep** — grid prep_sem with best identity weights; rank by ``stage2_*`` (prep_score).

  Saves per-config match CSVs plus identity/prep leaderboards. Final blended score is
  reported as ``final_*`` but not used for HP ranking.
    """
    hp_dir = hp_sweep_dir(work_dir)
    hp_dir.mkdir(parents=True, exist_ok=True)
    prep_grid = prep_grid if prep_grid is not None else identity_grid
    base = base_config or StagedMatchConfig()

    identity_configs = list(iter_identity_configs(identity_grid, base_config=base))
    n_id = len(identity_configs)
    print(f"Phase 1 — identity grid: {n_id} configs (rank by {STAGE1_RANK_KEY})", flush=True)

    identity_rows: list[dict[str, Any]] = []
    id_iter: Any = identity_configs
    if show_progress:
        tqdm_cls = _tqdm_factory()
        if tqdm_cls:
            id_iter = tqdm_cls(
                identity_configs,
                **_tqdm_kwargs(
                    total=n_id,
                    desc="HP identity",
                    leave=True,
                    position=None,
                    unit=None,
                    tqdm_cls=tqdm_cls,
                ),
            )

    for run_index, config in enumerate(id_iter, start=1):
        out_path = matches_path_for_config(hp_dir, config, phase="identity")
        matches = _run_one_config(
            config,
            parsed_ingredients=parsed_ingredients,
            name_embeddings=name_embeddings,
            prep_embeddings=prep_embeddings,
            dequant_embeddings=dequant_embeddings,
            food_index=food_index,
            out_path=out_path,
            force=force,
            show_progress=show_progress,
            progress_desc=f"Identity {config_slug(config)}",
        )
        metrics = summarize_staged_match_metrics(matches)
        row = _metrics_row(
            config,
            metrics,
            phase="identity",
            run_index=run_index,
            total_runs=n_id,
            matches_path=out_path,
            rank_key=STAGE1_RANK_KEY,
        )
        _print_staged_row(row, phase="identity")
        identity_rows.append(row)

    identity_df = pd.DataFrame(identity_rows).sort_values(STAGE1_RANK_KEY, ascending=False)
    identity_path = hp_dir / HP_IDENTITY_LEADERBOARD
    identity_df.to_csv(identity_path, index=False)
    print(f"\nIdentity leaderboard → {identity_path}", flush=True)

    best_identity_row = identity_df.iloc[0]
    identity_config = _config_from_identity_row(best_identity_row, base)

    prep_configs = list(iter_prep_configs(prep_grid, identity_config))
    n_prep = len(prep_configs)
    print(
        f"\nPhase 2 — prep grid: {n_prep} configs (rank by {STAGE2_RANK_KEY}); "
        f"identity={config_slug(identity_config)}",
        flush=True,
    )

    prep_rows: list[dict[str, Any]] = []
    prep_iter: Any = prep_configs
    if show_progress:
        tqdm_cls = _tqdm_factory()
        if tqdm_cls:
            prep_iter = tqdm_cls(
                prep_configs,
                **_tqdm_kwargs(
                    total=n_prep,
                    desc="HP prep",
                    leave=True,
                    position=None,
                    unit=None,
                    tqdm_cls=tqdm_cls,
                ),
            )

    for run_index, config in enumerate(prep_iter, start=1):
        out_path = matches_path_for_config(hp_dir, config, phase="prep")
        matches = _run_one_config(
            config,
            parsed_ingredients=parsed_ingredients,
            name_embeddings=name_embeddings,
            prep_embeddings=prep_embeddings,
            dequant_embeddings=dequant_embeddings,
            food_index=food_index,
            out_path=out_path,
            force=force,
            show_progress=show_progress,
            progress_desc=f"Prep {config_slug(config)}",
        )
        metrics = summarize_staged_match_metrics(matches)
        row = _metrics_row(
            config,
            metrics,
            phase="prep",
            run_index=run_index,
            total_runs=n_prep,
            matches_path=out_path,
            rank_key=STAGE2_RANK_KEY,
        )
        _print_staged_row(row, phase="prep")
        prep_rows.append(row)

    prep_df = pd.DataFrame(prep_rows).sort_values(STAGE2_RANK_KEY, ascending=False)
    prep_path = hp_dir / HP_PREP_LEADERBOARD
    prep_df.to_csv(prep_path, index=False)
    print(f"\nPrep leaderboard → {prep_path}", flush=True)

    best_prep_row = prep_df.iloc[0]
    prep_core = float(best_prep_row["prep_semantic_weight"]) + float(
        best_prep_row["prep_lexical_weight"]
    )
    prep_sem_frac = (
        float(best_prep_row["prep_semantic_weight"]) / prep_core if prep_core > 0 else 0.5
    )
    best_config = base.with_lexical_semantic_grid(
        name_semantic=float(best_prep_row["base_name_semantic_weight"]),
        dequant_semantic=float(best_prep_row["base_dequant_semantic_weight"]),
        prep_semantic=prep_sem_frac,
    )

    best_payload = {
        "config_slug": config_slug(best_config),
        "identity_leaderboard": str(identity_path.name),
        "prep_leaderboard": str(prep_path.name),
        "best_matches_file": str(best_prep_row["matches_file"]),
        "stage1_rank_metric": STAGE1_RANK_KEY,
        "stage2_rank_metric": STAGE2_RANK_KEY,
        "best_stage1_avg": best_identity_row.get(STAGE1_RANK_KEY),
        "best_stage2_avg": best_prep_row.get(STAGE2_RANK_KEY),
        **best_config.hp_fields(),
    }
    best_json_path = hp_dir / HP_BEST_CONFIG_JSON
    best_json_path.write_text(json.dumps(best_payload, indent=2) + "\n")
    print(f"Best config → {best_json_path}", flush=True)

    return {
        "identity_leaderboard": identity_df,
        "prep_leaderboard": prep_df,
        "best_config": best_config,
        "best_payload": best_payload,
        "hp_dir": hp_dir,
    }


# Back-compat: full 3D grid in one pass (still uses staged metrics columns)
def run_hp_sweep(
    parsed_ingredients: pd.DataFrame,
    name_embeddings: np.ndarray,
    prep_embeddings: np.ndarray,
    dequant_embeddings: np.ndarray,
    food_index: StagedFoodIndex,
    **kwargs: Any,
) -> pd.DataFrame:
    """Deprecated: use run_staged_hp_grid_search for sequential staged evaluation."""
    result = run_staged_hp_grid_search(
        parsed_ingredients,
        name_embeddings,
        prep_embeddings,
        dequant_embeddings,
        food_index,
        **kwargs,
    )
    return pd.concat(
        [result["identity_leaderboard"], result["prep_leaderboard"]],
        ignore_index=True,
    )


def load_embedding_artifacts(
    work_dir: Path,
    *,
    recipe_csv: Path,
    recipe_nrows: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, StagedFoodIndex]:
    recipe_ingredients = load_recipe_ingredients(recipe_csv=recipe_csv, nrows=recipe_nrows)
    parsed, name_emb, prep_emb, dequant_emb, _ = load_or_build_recipe_artifacts(
        recipe_ingredients, work_dir
    )
    food_df = load_food_4macro()
    _, food_name, food_prep, food_dequant, _ = load_or_build_food_artifacts(food_df, work_dir)
    food_index = StagedFoodIndex.from_catalog(
        food_df,
        name_embeddings=food_name,
        prep_embeddings=food_prep,
        dequant_embeddings=food_dequant,
        show_progress=True,
    )
    return parsed, name_emb, prep_emb, dequant_emb, food_df, food_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--recipe-csv",
        type=Path,
        default=ROOT / "Data" / "recipes" / "RecipeNLG.csv",
    )
    parser.add_argument("--recipe-nrows", type=int, default=10_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    grid = QUICK_SEMANTIC_GRID if args.quick else DEFAULT_SEMANTIC_GRID
    parsed, name_emb, prep_emb, dequant_emb, _, food_index = load_embedding_artifacts(
        args.work_dir,
        recipe_csv=args.recipe_csv,
        recipe_nrows=args.recipe_nrows,
    )
    run_staged_hp_grid_search(
        parsed,
        name_emb,
        prep_emb,
        dequant_emb,
        food_index,
        work_dir=args.work_dir,
        identity_grid=grid,
        prep_grid=grid,
        force=args.force,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
