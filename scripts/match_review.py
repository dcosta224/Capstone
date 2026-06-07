"""Filter and inspect ingredient→food match results for error analysis."""

from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

MatchQuality = Literal["high", "medium", "low", "unresolved"]
MatchStage = Literal["exact", "token_fuzzy", "unresolved"]

REVIEW_COLUMNS = [
    "recipe_id",
    "ingredient_idx",
    "ingredient",
    "name",
    "match_query",
    "matched_fdc_id",
    "matched_description",
    "match_quality",
    "match_stage",
    "match_score",
    "match_margin",
    "token_recall",
    "fuzzy_score",
    "exact_hit",
    "n_candidates",
    "parse_status",
]

SEARCH_COLUMNS = (
    "ingredient",
    "name",
    "match_query",
    "matched_description",
)


def _as_list(value: str | int | Sequence[str | int] | None) -> list | None:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return [value]
    return list(value)


def filter_matches(
    matches: pd.DataFrame,
    *,
    quality: str | Sequence[str] | None = None,
    exclude_quality: str | Sequence[str] | None = None,
    stage: str | Sequence[str] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    min_margin: float | None = None,
    max_margin: float | None = None,
    recipe_id: int | Sequence[int] | None = None,
    search: str | None = None,
    parse_status: str | Sequence[str] | None = None,
    has_fdc_id: bool | None = None,
    ner_mismatch_only: bool = False,
) -> pd.DataFrame:
    """Return rows matching error-analysis filters (does not copy the full frame)."""
    out = matches

    q = _as_list(quality)
    if q is not None:
        out = out.loc[out["match_quality"].isin(q)]

    ex = _as_list(exclude_quality)
    if ex is not None:
        out = out.loc[~out["match_quality"].isin(ex)]

    st = _as_list(stage)
    if st is not None:
        out = out.loc[out["match_stage"].isin(st)]

    if min_score is not None:
        out = out.loc[out["match_score"] >= min_score]
    if max_score is not None:
        out = out.loc[out["match_score"] <= max_score]
    if min_margin is not None:
        out = out.loc[out["match_margin"] >= min_margin]
    if max_margin is not None:
        out = out.loc[out["match_margin"] <= max_margin]

    rids = _as_list(recipe_id)
    if rids is not None:
        out = out.loc[out["recipe_id"].isin(rids)]

    ps = _as_list(parse_status)
    if ps is not None:
        out = out.loc[out["parse_status"].isin(ps)]

    if has_fdc_id is True:
        out = out.loc[out["matched_fdc_id"].notna()]
    elif has_fdc_id is False:
        out = out.loc[out["matched_fdc_id"].isna()]

    if search:
        needle = search.casefold()
        mask = pd.Series(False, index=out.index)
        for col in SEARCH_COLUMNS:
            if col in out.columns:
                mask |= out[col].fillna("").astype(str).str.casefold().str.contains(
                    needle, regex=False
                )
        out = out.loc[mask]

    if ner_mismatch_only and "ner_match" in out.columns:
        out = out.loc[out["ner_match"].eq(False)]

    return out


def review_matches(
    matches: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sort_by: str = "match_score",
    ascending: bool = True,
    limit: int | None = 50,
    random: bool = False,
    seed: int = 0,
    **filters,
) -> pd.DataFrame:
    """Filter matches and return a compact, sortable view for manual review."""
    subset = filter_matches(matches, **filters)
    cols = [c for c in (columns or REVIEW_COLUMNS) if c in subset.columns]
    view = subset[cols].copy()

    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=ascending, kind="stable")

    if random:
        n = min(limit or len(view), len(view))
        view = view.sample(n=n, random_state=seed)
    elif limit is not None:
        view = view.head(limit)

    return view.reset_index(drop=True)


def summarize_match_qualities(matches: pd.DataFrame) -> pd.DataFrame:
    """Counts and score ranges per match_quality tier."""
    rows = []
    for quality in ("high", "medium", "low", "unresolved"):
        tier = matches.loc[matches["match_quality"] == quality]
        if tier.empty:
            rows.append(
                {
                    "match_quality": quality,
                    "count": 0,
                    "pct": 0.0,
                    "match_score_min": None,
                    "match_score_median": None,
                    "match_score_max": None,
                    "match_margin_median": None,
                }
            )
            continue
        rows.append(
            {
                "match_quality": quality,
                "count": len(tier),
                "pct": round(100 * len(tier) / len(matches), 2),
                "match_score_min": round(tier["match_score"].min(), 4),
                "match_score_median": round(tier["match_score"].median(), 4),
                "match_score_max": round(tier["match_score"].max(), 4),
                "match_margin_median": round(tier["match_margin"].median(), 4),
            }
        )
    return pd.DataFrame(rows)


def load_matches_csv(path: str | pd.PathLike) -> pd.DataFrame:
    """Load saved ingredient_matches CSV from scratch/."""
    return pd.read_csv(path, dtype={"recipe_id": "int64", "ingredient_idx": "int64"})
