"""Per-recipe aggregates from ingredient-level match results."""

from __future__ import annotations

import pandas as pd


def summarize_recipe_matches(
    ingredient_matches: pd.DataFrame,
    recipes: pd.DataFrame,
    *,
    recipe_id_col: str = "recipe_id",
) -> pd.DataFrame:
    """One row per recipe with quality-tier percentages and derived metrics."""
    quality_counts = (
        ingredient_matches.groupby([recipe_id_col, "match_quality"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for col in ("high", "medium", "low", "unresolved"):
        if col not in quality_counts.columns:
            quality_counts[col] = 0

    summary = (
        ingredient_matches.groupby(recipe_id_col, as_index=False)
        .agg(
            n_ingredients=("match_quality", "count"),
            avg_match_score=("match_score", "mean"),
        )
        .merge(quality_counts.reset_index(), on=recipe_id_col, how="left")
    )

    summary = (
        recipes[[recipe_id_col]]
        .drop_duplicates()
        .merge(summary, on=recipe_id_col, how="left")
    )

    summary["n_ingredients"] = summary["n_ingredients"].fillna(0).astype(int)
    for col in ("high", "medium", "low", "unresolved"):
        summary[col] = summary[col].fillna(0).astype(int)

    has_ing = summary["n_ingredients"] > 0
    n = summary["n_ingredients"].where(has_ing)
    summary["percent_high"] = (100 * summary["high"] / n).where(has_ing).round(2)
    summary["percent_medium"] = (100 * summary["medium"] / n).where(has_ing).round(2)
    summary["percent_low"] = (100 * summary["low"] / n).where(has_ing).round(2)
    summary["percent_unmatched"] = (100 * summary["unresolved"] / n).where(has_ing).round(2)
    summary["avg_match_score"] = summary["avg_match_score"].where(has_ing).round(4)

    # Derived: share at medium-or-above; flag recipes where every ingredient is high
    summary["percent_med_high"] = (
        summary["percent_high"].fillna(0) + summary["percent_medium"].fillna(0)
    ).round(2)
    summary["percent_high_100"] = (
        (summary["percent_high"] == 100).where(has_ing).astype(float) * 100
    ).round(2)

    return summary[
        [
            recipe_id_col,
            "n_ingredients",
            "percent_high",
            "percent_medium",
            "percent_low",
            "percent_unmatched",
            "percent_med_high",
            "percent_high_100",
            "avg_match_score",
            "high",
            "medium",
            "low",
            "unresolved",
        ]
    ]


def _score_column_metrics(
    series: pd.Series,
    prefix: str,
    *,
    thresholds: tuple[float, ...] = (0.40, 0.55, 0.75),
) -> dict[str, float | int | None]:
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {f"{prefix}_avg": None, f"{prefix}_median": None, f"{prefix}_n": 0}
    out: dict[str, float | int | None] = {
        f"{prefix}_n": n,
        f"{prefix}_avg": round(float(s.mean()), 4),
        f"{prefix}_median": round(float(s.median()), 4),
        f"{prefix}_std": round(float(s.std()), 4) if n > 1 else 0.0,
    }
    for t in thresholds:
        key = f"{prefix}_pct_gte_{str(t).replace('.', '_')}"
        out[key] = round(100 * float((s >= t).mean()), 2)
    return out


def quality_tiers_from_scores(
    scores: pd.Series,
    *,
    high: float = 0.75,
    medium: float = 0.55,
    low: float = 0.40,
) -> dict[str, float]:
    """Tier percentages for a single score column (independent of final blend)."""
    s = scores.dropna()
    n = len(s)
    if n == 0:
        return {tier: 0.0 for tier in ("high", "medium", "low", "unresolved")}

    def tier(row: float) -> str:
        if row >= high:
            return "high"
        if row >= medium:
            return "medium"
        if row >= low:
            return "low"
        return "unresolved"

    vc = s.map(tier).value_counts()
    return {t: round(100 * int(vc.get(t, 0)) / n, 2) for t in ("high", "medium", "low", "unresolved")}


def summarize_staged_match_metrics(ingredient_matches: pd.DataFrame) -> dict[str, float | int | None]:
    """
    Staged evaluation — separate aggregates per stage score column.

    Use stage1_* (base_score) to rank identity HP, stage2_* (prep_score) for prep HP.
    final_* is the blended match_score for reference only.
    """
    n = len(ingredient_matches)
    if n == 0:
        return {"n_ingredients": 0}

    resolved = ingredient_matches["matched_fdc_id"].notna()
    out: dict[str, float | int | None] = {
        "n_ingredients": n,
        "resolved_pct": round(100 * float(resolved.mean()), 2),
    }

    if "base_score" in ingredient_matches.columns:
        out.update(_score_column_metrics(ingredient_matches["base_score"], "stage1"))
        for tier, pct in quality_tiers_from_scores(ingredient_matches["base_score"]).items():
            out[f"stage1_quality_{tier}_pct"] = pct

    if "prep_score" in ingredient_matches.columns:
        out.update(_score_column_metrics(ingredient_matches["prep_score"], "stage2"))
        for tier, pct in quality_tiers_from_scores(ingredient_matches["prep_score"]).items():
            out[f"stage2_quality_{tier}_pct"] = pct

    if "match_score" in ingredient_matches.columns:
        out.update(_score_column_metrics(ingredient_matches["match_score"], "final"))
        for tier, pct in quality_tiers_from_scores(ingredient_matches["match_score"]).items():
            out[f"final_quality_{tier}_pct"] = pct

    if "name_channel_score" in ingredient_matches.columns:
        out.update(
            _score_column_metrics(ingredient_matches["name_channel_score"], "name_channel")
        )
    if "dequant_channel_score" in ingredient_matches.columns:
        out.update(
            _score_column_metrics(ingredient_matches["dequant_channel_score"], "dequant_channel")
        )

    return out


def summarize_ingredient_matches(ingredient_matches: pd.DataFrame) -> pd.Series:
    """Aggregate counts, tier percentages, and mean match_score over ingredient lines."""
    n = len(ingredient_matches)
    if n == 0:
        return pd.Series(dtype=float)

    vc = ingredient_matches["match_quality"].value_counts()
    out: dict[str, float | int] = {"n_ingredients": n}
    for tier in ("high", "medium", "low", "unresolved"):
        count = int(vc.get(tier, 0))
        out[f"{tier}_count"] = count
        out[f"{tier}_pct"] = round(100 * count / n, 2)
    out["avg_match_score"] = round(ingredient_matches["match_score"].mean(), 4)
    resolved = ingredient_matches["matched_fdc_id"].notna()
    if resolved.any():
        out["avg_match_score_resolved"] = round(
            ingredient_matches.loc[resolved, "match_score"].mean(), 4
        )
    else:
        out["avg_match_score_resolved"] = None
    out["resolved_pct"] = round(100 * resolved.sum() / n, 2)
    return pd.Series(out)


def compare_ingredient_match_runs(
    mvp_matches: pd.DataFrame,
    full_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side ingredient-level tier counts, percentages, and scores."""
    mvp = summarize_ingredient_matches(mvp_matches)
    full = summarize_ingredient_matches(full_matches)
    table = pd.DataFrame({"food_mvp": mvp, "full_catalog": full})
    table["delta_full_minus_mvp"] = (table["full_catalog"] - table["food_mvp"]).round(4)
    return table


def compare_match_summaries(
    mvp_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    *,
    recipe_id_col: str = "recipe_id",
) -> pd.DataFrame:
    """Side-by-side recipe metrics for food_mvp vs full catalog matching."""
    m = mvp_summary.add_prefix("mvp_").rename(columns={f"mvp_{recipe_id_col}": recipe_id_col})
    f = full_summary.add_prefix("full_").rename(columns={f"full_{recipe_id_col}": recipe_id_col})
    merged = m.merge(f, on=recipe_id_col, how="outer")

    metric_cols = [
        "percent_high",
        "percent_medium",
        "percent_low",
        "percent_unmatched",
        "percent_med_high",
        "percent_high_100",
        "avg_match_score",
    ]
    for col in metric_cols:
        merged[f"delta_{col}"] = (merged[f"full_{col}"] - merged[f"mvp_{col}"]).round(4)

    return merged


def recipes_all_but_one_resolved_either_catalog(
    mvp_matches: pd.DataFrame,
    full_matches: pd.DataFrame,
    recipes: pd.DataFrame | None = None,
    *,
    recipe_id_col: str = "recipe_id",
    ingredient_idx_col: str = "ingredient_idx",
) -> tuple[float, pd.DataFrame]:
    """
    Recipes where at most one ingredient line lacks a match in *either* food_mvp or full catalog.

    Returns (pct_of_recipes_with_ingredients, detail_df one row per qualifying recipe).
    """
    keys = [recipe_id_col, ingredient_idx_col]
    m = mvp_matches[keys + ["matched_fdc_id"]].rename(
        columns={"matched_fdc_id": "mvp_fdc_id"}
    )
    f = full_matches[keys + ["matched_fdc_id"]].rename(
        columns={"matched_fdc_id": "full_fdc_id"}
    )
    both = m.merge(f, on=keys, how="inner")
    either = both["mvp_fdc_id"].notna() | both["full_fdc_id"].notna()

    per_recipe = (
        both.assign(resolved_either=either)
        .groupby(recipe_id_col, as_index=False)
        .agg(
            n_ingredients=(ingredient_idx_col, "count"),
            n_resolved_either=("resolved_either", "sum"),
        )
    )
    per_recipe["n_unresolved_either"] = (
        per_recipe["n_ingredients"] - per_recipe["n_resolved_either"]
    )
    qualifying = per_recipe.loc[per_recipe["n_unresolved_either"] <= 1].copy()

    with_ing = per_recipe.loc[per_recipe["n_ingredients"] > 0]
    rate = 100 * len(qualifying) / len(with_ing) if len(with_ing) else 0.0

    if recipes is not None and "title" in recipes.columns:
        qualifying = qualifying.merge(
            recipes[[recipe_id_col, "title"]].drop_duplicates(),
            on=recipe_id_col,
            how="left",
        )

    return rate, qualifying.sort_values(
        ["n_unresolved_either", "n_ingredients"], ascending=[True, False]
    ).reset_index(drop=True)


def ingredient_match_comparison(
    mvp_matches: pd.DataFrame,
    full_matches: pd.DataFrame,
    *,
    recipe_id_col: str = "recipe_id",
    ingredient_idx_col: str = "ingredient_idx",
) -> dict[str, float | int]:
    """Aggregate ingredient-level agreement between two match runs."""
    keys = [recipe_id_col, ingredient_idx_col]
    m = mvp_matches[keys + ["matched_fdc_id", "match_quality", "match_score"]].rename(
        columns={
            "matched_fdc_id": "mvp_fdc_id",
            "match_quality": "mvp_quality",
            "match_score": "mvp_score",
        }
    )
    f = full_matches[keys + ["matched_fdc_id", "match_quality", "match_score"]].rename(
        columns={
            "matched_fdc_id": "full_fdc_id",
            "match_quality": "full_quality",
            "match_score": "full_score",
        }
    )
    both = m.merge(f, on=keys, how="inner")
    n = len(both)
    mvp_res = both["mvp_fdc_id"].notna()
    full_res = both["full_fdc_id"].notna()
    both_res = mvp_res & full_res
    same_fdc = both_res & (both["mvp_fdc_id"] == both["full_fdc_id"])
    return {
        "n_ingredients": n,
        "mvp_resolved_pct": round(100 * mvp_res.sum() / n, 2) if n else 0.0,
        "full_resolved_pct": round(100 * full_res.sum() / n, 2) if n else 0.0,
        "both_resolved_pct": round(100 * both_res.sum() / n, 2) if n else 0.0,
        "same_fdc_when_both_pct": round(100 * same_fdc.sum() / both_res.sum(), 2)
        if both_res.any()
        else 0.0,
        "full_higher_score_pct": round(
            100 * (both["full_score"] > both["mvp_score"]).sum() / n, 2
        )
        if n
        else 0.0,
    }
