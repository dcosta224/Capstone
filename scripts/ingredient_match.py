"""Match parsed recipe ingredients to USDA food catalog rows."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from food_name_prefixes import words_before_delimiters

STOPWORDS = frozenset(
    """
    a an and as at by for from in into of on or per the to with without
    optional desired taste garnish serving
    """.split()
)

CONTAINER_WORDS = frozenset(
    """
    pkg pkg. package can cans jar jars bottle bottles box boxes bag bags
    carton stick sticks slice slices piece pieces wedge wedges
    """.split()
)

HIGH_FREQ_TOKENS = frozenset(
    """
    chicken milk sugar salt water oil cheese beef butter egg eggs flour
    onion garlic pepper sauce cream corn rice potato potatoes tomato
    """.split()
)

PREP_HINTS = frozenset(
    """
    chopped diced minced sliced grated shredded crushed ground peeled
    cooked raw fresh frozen canned drained softened melted roasted
    boiled baked fried steamed cut cubed
    """.split()
)

SCORE_HIGH = 0.75
SCORE_MEDIUM = 0.55
SCORE_LOW = 0.40
MARGIN_MIN = 0.08


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def content_tokens(value: Any, *, drop_containers: bool = True) -> set[str]:
    tokens = set(normalize_text(value).split()) - STOPWORDS
    if drop_containers:
        tokens -= CONTAINER_WORDS
    return {t for t in tokens if len(t) > 1 or t.isdigit()}


def prep_tokens(value: Any) -> set[str]:
    return content_tokens(value, drop_containers=False) & PREP_HINTS


def _field_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def parsed_name(row: pd.Series) -> str:
    """Primary match text: ingredient-parser `name`, else raw ingredient line."""
    name_s = _field_text(row.get("name"))
    if name_s:
        return name_s
    return _field_text(row.get("ingredient"))


def parsed_preparation(row: pd.Series) -> str:
    return _field_text(row.get("preparation"))


def parsed_size(row: pd.Series) -> str | None:
    size_s = _field_text(row.get("size"))
    return size_s or None


def build_query_text(row: pd.Series) -> str:
    """Logged match query (name only; prep/size scored separately)."""
    return parsed_name(row)


@dataclass
class FoodRecord:
    fdc_id: int
    data_type: str
    description: str
    ingredient: str
    prefix: str
    name: str | None
    size: str | None
    preparation: str | None
    norm_description: str
    norm_name: str
    norm_prefix: str
    tokens: set[str] = field(default_factory=set)


@dataclass
class FoodMatcher:
    records: dict[int, FoodRecord]
    token_index: dict[str, set[int]]
    max_candidates: int = 30

    @classmethod
    def from_dataframe(cls, food_df: pd.DataFrame, *, max_candidates: int = 30) -> FoodMatcher:
        records: dict[int, FoodRecord] = {}
        token_index: dict[str, set[int]] = defaultdict(set)

        for row in food_df.itertuples(index=False):
            fdc_id = int(row.fdc_id)
            desc = str(row.description) if row.description is not None else ""
            name = getattr(row, "name", None)
            name_s = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name)
            prefix = words_before_delimiters(desc)
            rec = FoodRecord(
                fdc_id=fdc_id,
                data_type=str(row.data_type),
                description=desc,
                ingredient=desc,
                prefix=prefix,
                name=name_s or None,
                size=getattr(row, "size", None),
                preparation=getattr(row, "preparation", None),
                norm_description=normalize_text(desc),
                norm_name=normalize_text(name_s),
                norm_prefix=normalize_text(prefix),
                tokens=content_tokens(desc) | content_tokens(name_s) | content_tokens(prefix),
            )
            records[fdc_id] = rec
            for tok in rec.tokens:
                token_index[tok].add(fdc_id)

        return cls(
            records=records,
            token_index=dict(token_index),
            max_candidates=max_candidates,
        )

    def block_candidates(self, query_tokens: set[str], max_candidates: int | None = None) -> list[int]:
        if max_candidates is None:
            max_candidates = self.max_candidates
        if not query_tokens:
            return []
        counts: Counter[int] = Counter()
        for tok in query_tokens:
            for fdc_id in self.token_index.get(tok, ()):
                counts[fdc_id] += 1
        if not counts:
            return []
        ranked = [fdc for fdc, _ in counts.most_common(max_candidates)]
        if len(query_tokens) == 1 and query_tokens & HIGH_FREQ_TOKENS and len(ranked) > max_candidates:
            return ranked[:max_candidates]
        return ranked

    def score_pair(
        self,
        query_name: str,
        name_tokens: set[str],
        query_prep: set[str],
        query_size: str | None,
        rec: FoodRecord,
    ) -> dict[str, float]:
        desc_tokens = rec.tokens
        overlap = name_tokens & desc_tokens
        token_recall = len(overlap) / len(name_tokens) if name_tokens else 0.0
        token_precision = len(overlap) / len(desc_tokens) if desc_tokens else 0.0

        # Fuzzy: parsed name vs food fields (name > prefix > full description).
        fuzzy_vs_name = fuzz.token_set_ratio(query_name, rec.name or "") / 100.0
        fuzzy_vs_prefix = fuzz.token_set_ratio(query_name, rec.prefix) / 100.0
        fuzzy_vs_desc = fuzz.token_set_ratio(query_name, rec.description) / 100.0
        fuzzy_score = max(
            fuzzy_vs_name,
            fuzzy_vs_prefix * 0.92,
            fuzzy_vs_desc * 0.75,
        )

        norm_name = normalize_text(query_name)
        exact_hit = float(
            norm_name
            and (
                norm_name == rec.norm_name
                or norm_name == rec.norm_prefix
                or norm_name in rec.norm_description
                or rec.norm_prefix in norm_name
                or rec.norm_name in norm_name
            )
        )

        size_match = 0.0
        if query_size:
            size_n = normalize_text(query_size)
            if size_n and size_n in rec.norm_description:
                size_match = 1.0

        prep_overlap = 0.0
        prep_mismatch = 0.0
        prep_fuzzy = 0.0
        if query_prep:
            rec_prep = prep_tokens(rec.preparation or "") | prep_tokens(rec.description)
            if query_prep & rec_prep:
                prep_overlap = 1.0
            else:
                prep_mismatch = 1.0
            prep_text = " ".join(sorted(query_prep))
            prep_fuzzy = fuzz.token_set_ratio(prep_text, rec.description) / 100.0

        # Weights: name (primary) > preparation (secondary) > size (tertiary).
        score = (
            0.36 * token_recall
            + 0.08 * token_precision
            + 0.36 * fuzzy_score
            + 0.12 * exact_hit
            + 0.06 * prep_overlap
            + 0.04 * prep_fuzzy
            + 0.04 * size_match
            - 0.10 * prep_mismatch
        )
        score = max(0.0, min(1.0, score))

        return {
            "token_recall": token_recall,
            "token_precision": token_precision,
            "fuzzy_score": fuzzy_score,
            "fuzzy_name": fuzzy_vs_name,
            "fuzzy_desc": fuzzy_vs_desc,
            "fuzzy_prefix": fuzzy_vs_prefix,
            "exact_hit": exact_hit,
            "size_match": size_match,
            "prep_overlap": prep_overlap,
            "prep_mismatch": prep_mismatch,
            "prep_fuzzy": prep_fuzzy,
            "match_score": score,
        }

    def match_row(self, row: pd.Series) -> dict[str, Any]:
        query_name = parsed_name(row)
        name_tokens = content_tokens(query_name)
        query_prep = prep_tokens(parsed_preparation(row))
        query_size = parsed_size(row)

        base = {
            "match_query": query_name,
            "matched_fdc_id": None,
            "matched_description": None,
            "match_stage": "unresolved",
            "match_score": 0.0,
            "match_margin": 0.0,
            "match_quality": "unresolved",
            "match_quality_score": 0.0,
            "n_candidates": 0,
            "token_recall": 0.0,
            "token_precision": 0.0,
            "fuzzy_score": 0.0,
            "exact_hit": 0.0,
        }

        if not query_name.strip():
            return base

        norm_name = normalize_text(query_name)
        for rec in self.records.values():
            if norm_name and (
                norm_name == rec.norm_name
                or norm_name == rec.norm_prefix
                or norm_name == rec.norm_description
            ):
                return {
                    **base,
                    "matched_fdc_id": rec.fdc_id,
                    "matched_description": rec.description,
                    "match_stage": "exact",
                    "match_score": 1.0,
                    "match_margin": 1.0,
                    "match_quality": "high",
                    "match_quality_score": 1.0,
                    "n_candidates": 1,
                    "token_recall": 1.0,
                    "token_precision": 1.0,
                    "fuzzy_score": 1.0,
                    "exact_hit": 1.0,
                }

        block_tokens = name_tokens
        if query_prep:
            block_tokens = block_tokens | query_prep

        candidates = self.block_candidates(block_tokens)
        if not candidates and name_tokens:
            candidates = self.block_candidates(
                name_tokens - HIGH_FREQ_TOKENS, max_candidates=40
            )
        if not candidates:
            return base

        scored: list[tuple[int, dict[str, float]]] = []
        for fdc_id in candidates:
            metrics = self.score_pair(
                query_name, name_tokens, query_prep, query_size, self.records[fdc_id]
            )
            scored.append((fdc_id, metrics))
        scored.sort(key=lambda x: x[1]["match_score"], reverse=True)

        best_fdc, best = scored[0]
        second_score = scored[1][1]["match_score"] if len(scored) > 1 else 0.0
        margin = best["match_score"] - second_score

        if best["exact_hit"] >= 1.0:
            stage = "exact"
        elif best["match_score"] >= SCORE_LOW:
            stage = "token_fuzzy"
        else:
            stage = "unresolved"

        if best["match_score"] >= SCORE_HIGH and margin >= MARGIN_MIN:
            quality = "high"
        elif best["match_score"] >= SCORE_MEDIUM:
            quality = "medium"
        elif best["match_score"] >= SCORE_LOW:
            quality = "low"
        else:
            quality = "unresolved"
            stage = "unresolved"

        rec = self.records[best_fdc]
        return {
            **base,
            "matched_fdc_id": best_fdc,
            "matched_description": rec.description,
            "match_stage": stage,
            "match_score": best["match_score"],
            "match_margin": margin,
            "match_quality": quality,
            "match_quality_score": best["match_score"],
            "n_candidates": len(candidates),
            "token_recall": best["token_recall"],
            "token_precision": best["token_precision"],
            "fuzzy_score": best["fuzzy_score"],
            "exact_hit": best["exact_hit"],
        }


def match_ingredients_to_food(
    recipe_ingredients: pd.DataFrame,
    food_catalog: pd.DataFrame,
    *,
    show_progress: bool = False,
    max_candidates: int = 30,
) -> pd.DataFrame:
    """Return recipe_ingredients with match metadata columns appended."""
    matcher = FoodMatcher.from_dataframe(food_catalog, max_candidates=max_candidates)

    rows = []
    series = recipe_ingredients.itertuples(index=False)
    if show_progress:
        try:
            from tqdm import tqdm

            series = tqdm(series, total=len(recipe_ingredients), desc="Matching")
        except ImportError:
            pass

    for row in series:
        row_series = pd.Series(row._asdict())
        rows.append(matcher.match_row(row_series))

    match_df = pd.DataFrame(rows)
    return pd.concat([recipe_ingredients.reset_index(drop=True), match_df], axis=1)
