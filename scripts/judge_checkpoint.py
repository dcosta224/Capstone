"""Disk checkpoint merge for LLM judge rows (feasibility runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

MERGE_KEYS = ("recipe_id", "ingredient_idx")


def load_judge_checkpoint(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def completed_keys(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df.empty:
        return set()
    return {
        (int(r.recipe_id), int(r.ingredient_idx))
        for r in df.itertuples(index=False)
    }


def merge_judge_checkpoint(path: Path, new_rows: list[dict[str, Any]]) -> int:
    """Upsert judge rows by (recipe_id, ingredient_idx). Returns total rows on disk."""
    if not new_rows:
        return len(load_judge_checkpoint(path))
    incoming = pd.DataFrame(new_rows)
    existing = load_judge_checkpoint(path)
    if existing.empty:
        merged = incoming
    else:
        keys = set(
            zip(
                incoming["recipe_id"].astype(int),
                incoming["ingredient_idx"].astype(int),
                strict=True,
            )
        )
        keep = existing[
            ~existing.apply(
                lambda r: (int(r["recipe_id"]), int(r["ingredient_idx"])) in keys,
                axis=1,
            )
        ]
        merged = pd.concat([keep, incoming], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return len(merged)


def no_portion_rate(df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (no_portion_count, total, rate)."""
    if df.empty or "grams_status" not in df.columns:
        return 0, 0, 0.0
    total = len(df)
    nop = int((df["grams_status"] == "no_portion").sum())
    return nop, total, nop / max(total, 1)
