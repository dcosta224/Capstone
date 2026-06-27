"""Disk checkpoint merge for LLM judge rows (feasibility runs)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

MERGE_KEYS = ("recipe_id", "ingredient_idx", "split_part_idx")


def _split_part_idx_series(df: pd.DataFrame) -> pd.Series:
    if "split_part_idx" in df.columns:
        return df["split_part_idx"].fillna(0).astype(int)
    return pd.Series(0, index=df.index, dtype=int)


def row_merge_key(row: Any) -> tuple[int, int, int]:
    if hasattr(row, "recipe_id"):
        sp = int(getattr(row, "split_part_idx", 0) or 0)
        return (int(row.recipe_id), int(row.ingredient_idx), sp)
    sp = int(row.get("split_part_idx", 0) or 0)
    return (int(row["recipe_id"]), int(row["ingredient_idx"]), sp)


def completed_keys(df: pd.DataFrame) -> set[tuple[int, int, int]]:
    if df.empty:
        return set()
    sp = _split_part_idx_series(df)
    return {
        (int(rid), int(iidx), int(part))
        for rid, iidx, part in zip(df["recipe_id"], df["ingredient_idx"], sp, strict=True)
    }


def load_judge_checkpoint(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.is_file():
            return pd.read_parquet(tmp_path)
        raise


def merge_judge_checkpoint(path: Path, new_rows: list[dict[str, Any]]) -> int:
    """Upsert judge rows by (recipe_id, ingredient_idx, split_part_idx)."""
    if not new_rows:
        return len(load_judge_checkpoint(path))
    incoming = pd.DataFrame(new_rows)
    if "split_part_idx" not in incoming.columns:
        incoming["split_part_idx"] = 0
    existing = load_judge_checkpoint(path)
    if existing.empty:
        merged = incoming
    else:
        if "split_part_idx" not in existing.columns:
            existing = existing.copy()
            existing["split_part_idx"] = 0
        incoming_sp = _split_part_idx_series(incoming)
        keys = {
            (int(rid), int(iidx), int(part))
            for rid, iidx, part in zip(
                incoming["recipe_id"], incoming["ingredient_idx"], incoming_sp, strict=True
            )
        }
        keep = existing[
            ~existing.apply(lambda r: row_merge_key(r) in keys, axis=1)
        ]
        merged = pd.concat([keep, incoming], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    return len(merged)


def no_portion_rate(df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (no_portion_count, total, rate)."""
    if df.empty or "grams_status" not in df.columns:
        return 0, 0, 0.0
    total = len(df)
    nop = int((df["grams_status"] == "no_portion").sum())
    return nop, total, nop / max(total, 1)


def combine_judged_checkpoint(
    disk_path: Path | None,
    session_rows: list[dict[str, Any]] | None = None,
    *,
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge baseline, on-disk checkpoint, and in-session rows (last wins per line key)."""
    parts: list[pd.DataFrame] = []
    if baseline_df is not None and not baseline_df.empty:
        parts.append(baseline_df)
    if disk_path is not None and disk_path.is_file():
        parts.append(load_judge_checkpoint(disk_path))
    if session_rows:
        parts.append(pd.DataFrame(session_rows))
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    if merged.empty:
        return merged
    if "split_part_idx" not in merged.columns:
        merged["split_part_idx"] = 0
    else:
        merged["split_part_idx"] = merged["split_part_idx"].fillna(0).astype(int)
    return merged.drop_duplicates(subset=list(MERGE_KEYS), keep="last")


def fully_judged_resolution_stats(
    judged_df: pd.DataFrame,
    expected_lines_per_recipe: dict[int, int],
) -> dict[str, Any]:
    """Among recipes with every expected line judged, count those fully resolved.

    A line is *resolved* when it has a real fdc_id and non-null grams (see
  ``recipe_complete_rate.line_fully_resolved``).
    """
    if not expected_lines_per_recipe:
        return {
            "n_fully_judged": 0,
            "n_fully_judged_resolved": 0,
            "fully_judged_resolved_pct": 0.0,
            "n_recipes_total": 0,
        }
    from recipe_complete_rate import line_fully_resolved

    n_fully_judged = 0
    n_fully_judged_resolved = 0
    if judged_df.empty:
        return {
            "n_fully_judged": 0,
            "n_fully_judged_resolved": 0,
            "fully_judged_resolved_pct": 0.0,
            "n_recipes_total": len(expected_lines_per_recipe),
        }

    judged = judged_df.copy()
    judged["recipe_id"] = judged["recipe_id"].astype(int)

    for rid, expected in expected_lines_per_recipe.items():
        sub = judged[judged["recipe_id"] == int(rid)]
        if len(sub) < int(expected):
            continue
        n_fully_judged += 1
        if all(line_fully_resolved(row) for row in sub.itertuples(index=False)):
            n_fully_judged_resolved += 1

    pct = 100.0 * n_fully_judged_resolved / max(n_fully_judged, 1)
    return {
        "n_fully_judged": n_fully_judged,
        "n_fully_judged_resolved": n_fully_judged_resolved,
        "fully_judged_resolved_pct": round(pct, 1),
        "n_recipes_total": len(expected_lines_per_recipe),
    }


def format_fully_judged_resolution_suffix(stats: dict[str, Any]) -> str:
    """Console fragment for judge progress lines."""
    n_judged = stats["n_fully_judged"]
    if n_judged == 0:
        return " | fully-judged resolved n/a (0 recipes complete)"
    return (
        f" | fully-judged resolved {stats['fully_judged_resolved_pct']:.1f}% "
        f"({stats['n_fully_judged_resolved']}/{n_judged})"
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_judge_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one judge row as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def _read_judge_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compact_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    """Merge all JSONL rows into parquet by (recipe_id, ingredient_idx); last row wins."""
    rows = _read_judge_jsonl(jsonl_path)
    if not rows:
        return len(load_judge_checkpoint(parquet_path))
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["recipe_id"]), int(row["ingredient_idx"]))
        by_key[key] = row
    return merge_judge_checkpoint(parquet_path, list(by_key.values()))
