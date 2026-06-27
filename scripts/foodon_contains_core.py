"""Precompute FoodOn class → allergen/restriction contains flags from ontology ancestors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from diet_tags_core import load_diet_tags
from foodon_paths import FOODON_CONTAINS_CSV, FOODON_CONTAINS_PARQUET, FOODON_CONTAINS_SUMMARY

DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parents[1] / "data" / "foodon_contains_overrides.json"


def _descendants_of(
    ancestor_id: str,
    children: dict[str, list[str]],
) -> set[str]:
    """All nodes at or below ancestor_id in the FoodOn hierarchy."""
    out: set[str] = set()
    stack = [ancestor_id]
    while stack:
        node = stack.pop()
        if node in out:
            continue
        out.add(node)
        stack.extend(children.get(node, []))
    return out


def _load_overrides(path: Path | None) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    p = path or DEFAULT_OVERRIDES_PATH
    if not p.is_file():
        return {}, {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    force = {str(k): list(v) for k, v in (raw.get("force_contains") or {}).items()}
    clear = {str(k): list(v) for k, v in (raw.get("clear_contains") or {}).items()}
    return force, clear


def build_contains_lookup(
    foodon_index: Any,
    *,
    overrides_path: Path | None = None,
    foodon_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return wide DataFrame: foodon_id, label, contains_<slug> booleans for each diet_tags contains key.
    """
    registry = load_diet_tags()
    contains_slugs = sorted(registry.contains.keys())
    children = foodon_index.children

    # Precompute descendant sets per (slug, ancestor_id).
    slug_hits: dict[str, set[str]] = {slug: set() for slug in contains_slugs}
    ancestor_meta: list[dict[str, Any]] = []
    for slug, trigger in registry.contains.items():
        for anc_id in trigger.foodon_ancestors:
            if anc_id not in foodon_index.labels:
                ancestor_meta.append(
                    {"contains_slug": slug, "ancestor_id": anc_id, "status": "missing_in_index"}
                )
                continue
            desc = _descendants_of(anc_id, children)
            slug_hits[slug] |= desc
            ancestor_meta.append(
                {
                    "contains_slug": slug,
                    "ancestor_id": anc_id,
                    "ancestor_label": foodon_index.labels.get(anc_id, ""),
                    "descendant_count": len(desc),
                    "status": "ok",
                }
            )
        for exc_id in trigger.foodon_ancestors_exclude:
            if exc_id not in foodon_index.labels:
                ancestor_meta.append(
                    {
                        "contains_slug": slug,
                        "ancestor_id": exc_id,
                        "status": "exclude_missing_in_index",
                    }
                )
                continue
            exc = _descendants_of(exc_id, children)
            slug_hits[slug] -= exc
            ancestor_meta.append(
                {
                    "contains_slug": slug,
                    "ancestor_id": exc_id,
                    "ancestor_label": foodon_index.labels.get(exc_id, ""),
                    "descendant_count": len(exc),
                    "status": "exclude",
                }
            )

    force, clear = _load_overrides(overrides_path)

    rows: list[dict[str, Any]] = []
    node_ids = sorted(foodon_index.labels.keys())
    if foodon_only:
        node_ids = [nid for nid in node_ids if str(nid).startswith("FOODON_")]

    for node_id in node_ids:
        label = foodon_index.labels.get(node_id, "")
        row: dict[str, Any] = {"foodon_id": node_id, "label": label}
        for slug in contains_slugs:
            val = node_id in slug_hits[slug]
            if slug in force.get(node_id, []):
                val = True
            if slug in clear.get(node_id, []):
                val = False
            row[f"contains_{slug}"] = bool(val)
        rows.append(row)

    df = pd.DataFrame(rows)

    summary: dict[str, Any] = {
        "nodes": len(df),
        "contains_slugs": contains_slugs,
        "tagged_counts": {
            slug: int(df[f"contains_{slug}"].sum()) for slug in contains_slugs if f"contains_{slug}" in df.columns
        },
        "ancestor_roots": ancestor_meta,
        "overrides_path": str(overrides_path or DEFAULT_OVERRIDES_PATH),
        "overrides_applied": {"force": len(force), "clear": len(clear)},
    }
    return df, summary


def write_contains_table(df: pd.DataFrame, summary: dict[str, Any]) -> Path:
    FOODON_CONTAINS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(FOODON_CONTAINS_PARQUET, index=False)
        out = FOODON_CONTAINS_PARQUET
    except ImportError:
        df.to_csv(FOODON_CONTAINS_CSV, index=False)
        out = FOODON_CONTAINS_CSV
    FOODON_CONTAINS_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def lookup_contains(
    foodon_id: str,
    table: pd.DataFrame,
    *,
    contains_slugs: list[str] | None = None,
) -> set[str]:
    """Return contains slugs that are True for a foodon_id row."""
    sub = table[table["foodon_id"] == foodon_id]
    if sub.empty:
        return set()
    row = sub.iloc[0]
    slugs = contains_slugs or [c.replace("contains_", "") for c in table.columns if c.startswith("contains_")]
    return {slug for slug in slugs if bool(row.get(f"contains_{slug}", False))}


def load_contains_table() -> pd.DataFrame | None:
    """Load precomputed contains table from scratch/tag if present."""
    if FOODON_CONTAINS_PARQUET.is_file():
        return pd.read_parquet(FOODON_CONTAINS_PARQUET)
    if FOODON_CONTAINS_CSV.is_file():
        return pd.read_csv(FOODON_CONTAINS_CSV)
    return None
