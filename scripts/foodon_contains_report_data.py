"""Load FoodOn contains cache + LLM audit stats for reports and charts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from foodon_paths import FOODON_CONTAINS_CSV, FOODON_CONTAINS_SUMMARY

ROOT = Path(__file__).resolve().parents[1]
TAG_DIR = ROOT / "scratch" / "tag"
DEFAULT_REVIEW_CSV = TAG_DIR / "foodon_contains_llm_review.csv"


@dataclass
class FoodOnContainsReport:
    total_nodes: int
    tagged_nodes: int
    untagged_nodes: int
    contains_slugs: list[str]
    cache_counts: dict[str, int]
    ancestor_only_counts: dict[str, int]
    label_keyword_additions: dict[str, int]
    llm_review_rows: int
    llm_raw_suggestions: int
    llm_confirmed: int
    llm_rejected: int
    llm_no_suggestion: int
    llm_errors: int
    confirmed_slug_counts: dict[str, int]
    sample_confirmed: list[dict[str, str]] = field(default_factory=list)


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.len() > 0


def load_report(
    *,
    contains_csv: Path = FOODON_CONTAINS_CSV,
    summary_json: Path = FOODON_CONTAINS_SUMMARY,
    review_csv: Path = DEFAULT_REVIEW_CSV,
    sample_n: int = 5,
) -> FoodOnContainsReport:
    if not contains_csv.is_file():
        raise FileNotFoundError(
            f"Missing {contains_csv}. Run: uv run python scripts/build_foodon_contains_cache.py"
        )

    table = pd.read_csv(contains_csv)
    slug_cols = [c for c in table.columns if c.startswith("contains_")]
    slugs = [c.removeprefix("contains_") for c in slug_cols]
    tagged_mask = table[slug_cols].astype(bool).any(axis=1)
    total = len(table)
    tagged = int(tagged_mask.sum())

    summary: dict = {}
    if summary_json.is_file():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

    cache_counts = {slug: int(table[f"contains_{slug}"].astype(bool).sum()) for slug in slugs}
    ancestor_only = summary.get("tagged_counts", {})
    if summary.get("ancestor_only_counts"):
        ancestor_only = summary["ancestor_only_counts"]
    keyword_adds = summary.get("label_keyword_additions", {s: 0 for s in slugs})

    llm_review_rows = 0
    llm_raw = 0
    llm_confirmed = 0
    llm_rejected = 0
    llm_no_suggestion = 0
    llm_errors = 0
    confirmed_slug_counts: dict[str, int] = {}
    sample_confirmed: list[dict[str, str]] = []

    if review_csv.is_file():
        review = pd.read_csv(review_csv)
        llm_review_rows = len(review)
        raw_mask = _nonempty(review["llm_added"])
        conf_mask = _nonempty(review["ontology_confirmed"])
        rej_mask = _nonempty(review["ontology_rejected"])
        llm_raw = int(raw_mask.sum())
        llm_confirmed = int(conf_mask.sum())
        llm_rejected = int(rej_mask.sum())
        llm_no_suggestion = int((~raw_mask).sum())
        if "llm_error" in review.columns:
            llm_errors = int(review["llm_error"].astype(bool).sum())

        for val in review.loc[conf_mask, "ontology_confirmed"].fillna(""):
            for slug in str(val).split(","):
                slug = slug.strip()
                if slug:
                    confirmed_slug_counts[slug] = confirmed_slug_counts.get(slug, 0) + 1

        for row in review.loc[conf_mask].head(sample_n).itertuples(index=False):
            sample_confirmed.append(
                {
                    "foodon_id": str(row.foodon_id),
                    "label": str(row.label),
                    "confirmed": str(row.ontology_confirmed),
                    "rationale": str(getattr(row, "llm_rationale", "")),
                }
            )

    return FoodOnContainsReport(
        total_nodes=total,
        tagged_nodes=tagged,
        untagged_nodes=total - tagged,
        contains_slugs=slugs,
        cache_counts=cache_counts,
        ancestor_only_counts={s: int(ancestor_only.get(s, 0)) for s in slugs},
        label_keyword_additions={s: int(keyword_adds.get(s, 0)) for s in slugs},
        llm_review_rows=llm_review_rows,
        llm_raw_suggestions=llm_raw,
        llm_confirmed=llm_confirmed,
        llm_rejected=llm_rejected,
        llm_no_suggestion=llm_no_suggestion,
        llm_errors=llm_errors,
        confirmed_slug_counts=confirmed_slug_counts,
        sample_confirmed=sample_confirmed,
    )
