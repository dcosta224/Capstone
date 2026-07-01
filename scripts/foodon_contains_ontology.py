"""Ontology-aware validation for FoodOn contains tag suggestions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from diet_tags_core import contains_slugs_from_label, load_diet_tags
from foodon_contains_core import lookup_contains

TAXONOMY_LABEL_RE = re.compile(
    r"\b(?:eurocode|gs1 gpc|ccpr|efsa foodex2|variety packs?|tenderisers?|"
    r"mechanically separated meat \(msm\)|mixed species meat)\b",
    re.IGNORECASE,
)

CATEGORY_LABEL_RE = re.compile(
    r"\b(?:category|index|specification|code\)|\d{5}\s*-\s*[a-z/]+(?:shelf stable|frozen|perishable))",
    re.IGNORECASE,
)

PIECE_OF_MEAT_RE = re.compile(r"\bpiece of .+ meat\b", re.IGNORECASE)

# High-confidence LLM fast-path only for animal-product dimensions.
PIECE_OF_MEAT_SLUGS = frozenset({"fish", "shellfish", "red_meat", "pork", "poultry"})


@dataclass
class SlugOntologyVerdict:
    slug: str
    confirmed: bool
    ontology_score: float
    label_match: bool
    under_configured_ancestor: bool
    ancestor_tagged: bool
    sibling_support: float
    descendant_support: float
    taxonomy_penalty: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class NodeOntologyContext:
    node_id: str
    label: str
    parent_ids: list[str]
    parent_labels: list[str]
    sibling_ids: list[str]
    child_ids: list[str]
    ancestor_ids: list[str]


def _slug_set(node_id: str, table: pd.DataFrame, extra: dict[str, set[str]] | None) -> set[str]:
    base = lookup_contains(node_id, table)
    if extra and node_id in extra:
        base |= extra[node_id]
    return base


def build_node_context(foodon_index: Any, node_id: str) -> NodeOntologyContext:
    label = foodon_index.labels.get(node_id, "")
    parent_ids = list(foodon_index.parents.get(node_id, []))
    parent_labels = [foodon_index.labels.get(p, p) for p in parent_ids]
    sibling_ids: list[str] = []
    for parent in parent_ids:
        for sib in foodon_index.children.get(parent, []):
            if sib != node_id:
                sibling_ids.append(sib)
    child_ids = list(foodon_index.children.get(node_id, []))
    ancestor_ids = foodon_index.ancestry_path(node_id)
    return NodeOntologyContext(
        node_id=node_id,
        label=label,
        parent_ids=parent_ids,
        parent_labels=parent_labels,
        sibling_ids=sibling_ids,
        child_ids=child_ids,
        ancestor_ids=ancestor_ids,
    )


def _support_rate(peer_ids: list[str], slug: str, table: pd.DataFrame, extra: dict[str, set[str]] | None) -> float:
    if not peer_ids:
        return 0.0
    hits = sum(1 for pid in peer_ids if slug in _slug_set(pid, table, extra))
    return hits / len(peer_ids)


def evaluate_slug_suggestion(
    *,
    node_id: str,
    slug: str,
    label: str,
    llm_confidence: float,
    foodon_index: Any,
    contains_table: pd.DataFrame,
    llm_batch_tags: dict[str, set[str]] | None = None,
    min_confirm_score: float = 0.55,
) -> SlugOntologyVerdict:
    """Score one LLM contains suggestion using ontology + label evidence."""
    registry = load_diet_tags()
    trigger = registry.contains.get(slug)
    ctx = build_node_context(foodon_index, node_id)

    label_slugs = contains_slugs_from_label(label, registry)
    label_match = slug in label_slugs

    under_root = False
    if trigger:
        under_root = foodon_index.matches_any_ancestor(node_id, trigger.foodon_ancestors)

    ancestor_tagged = any(slug in _slug_set(a, contains_table, llm_batch_tags) for a in ctx.ancestor_ids)
    sibling_support = _support_rate(ctx.sibling_ids, slug, contains_table, llm_batch_tags)
    descendant_support = _support_rate(ctx.child_ids, slug, contains_table, llm_batch_tags)

    taxonomy_penalty = bool(TAXONOMY_LABEL_RE.search(label) or CATEGORY_LABEL_RE.search(label))

    score = 0.0
    reasons: list[str] = []

    if label_match:
        score += 0.35
        reasons.append("label_keyword")
    if under_root:
        score += 0.35
        reasons.append("under_configured_ancestor")
    if ancestor_tagged:
        score += 0.15
        reasons.append("ancestor_already_tagged")
    if sibling_support >= 0.2:
        score += min(0.25, sibling_support * 0.4)
        reasons.append(f"sibling_support={sibling_support:.2f}")
    if descendant_support >= 0.2:
        score += min(0.25, descendant_support * 0.4)
        reasons.append(f"descendant_support={descendant_support:.2f}")
    if llm_confidence >= 0.85:
        score += 0.1
        reasons.append("high_llm_confidence")

    if taxonomy_penalty:
        score -= 0.5
        reasons.append("taxonomy_category_penalty")

    piece_of_meat_leaf = _high_conf_piece_of_meat(label, slug, llm_confidence)
    if piece_of_meat_leaf:
        score = max(score, 0.65)
        reasons.append("piece_of_meat_high_conf")

    # Confirmation: need structural or label evidence, not LLM alone.
    has_structure = under_root or ancestor_tagged or sibling_support >= 0.25 or descendant_support >= 0.5
    has_label = label_match
    strong_llm_leaf = (
        llm_confidence >= 0.9
        and not taxonomy_penalty
        and not ctx.child_ids
        and (label_match or _looks_like_specific_food(label))
    )

    confirmed = (not taxonomy_penalty) and (
        piece_of_meat_leaf
        or (has_label and has_structure)
        or (has_label and llm_confidence >= 0.75)
        or (has_structure and llm_confidence >= 0.8)
        or (strong_llm_leaf and score >= min_confirm_score)
    )

    return SlugOntologyVerdict(
        slug=slug,
        confirmed=confirmed,
        ontology_score=round(max(0.0, min(1.0, score)), 3),
        label_match=label_match,
        under_configured_ancestor=under_root,
        ancestor_tagged=ancestor_tagged,
        sibling_support=round(sibling_support, 3),
        descendant_support=round(descendant_support, 3),
        taxonomy_penalty=taxonomy_penalty,
        reasons=reasons,
    )


def _high_conf_piece_of_meat(label: str, slug: str, llm_confidence: float) -> bool:
    """Confirm off-branch anatomy leaves when LLM is very confident."""
    if slug not in PIECE_OF_MEAT_SLUGS or llm_confidence < 0.9:
        return False
    if TAXONOMY_LABEL_RE.search(label) or CATEGORY_LABEL_RE.search(label):
        return False
    return bool(PIECE_OF_MEAT_RE.search(label))


def _looks_like_specific_food(label: str) -> bool:
    text = label.lower()
    if TAXONOMY_LABEL_RE.search(text) or CATEGORY_LABEL_RE.search(text):
        return False
    return bool(
        re.search(
            r"\b(?:piece of|food product|beverage|stew|broth|soup|cheese|milk|raw\)|cooked\))",
            text,
        )
    )


def evaluate_llm_suggestions(
    *,
    node_id: str,
    label: str,
    current: set[str],
    llm_contains: set[str],
    llm_confidence: float,
    foodon_index: Any,
    contains_table: pd.DataFrame,
    llm_batch_tags: dict[str, set[str]] | None = None,
    min_confirm_score: float = 0.55,
) -> tuple[list[str], list[str], list[SlugOntologyVerdict]]:
    """
    Return (confirmed_additions, rejected_additions, per-slug verdicts) for new LLM slugs only.
    """
    added = sorted(set(llm_contains) - set(current))
    confirmed: list[str] = []
    rejected: list[str] = []
    verdicts: list[SlugOntologyVerdict] = []

    for slug in added:
        v = evaluate_slug_suggestion(
            node_id=node_id,
            slug=slug,
            label=label,
            llm_confidence=llm_confidence,
            foodon_index=foodon_index,
            contains_table=contains_table,
            llm_batch_tags=llm_batch_tags,
            min_confirm_score=min_confirm_score,
        )
        verdicts.append(v)
        if v.confirmed:
            confirmed.append(slug)
        else:
            rejected.append(slug)

    return confirmed, rejected, verdicts


def format_ontology_summary(verdicts: list[SlugOntologyVerdict]) -> str:
    parts: list[str] = []
    for v in verdicts:
        flag = "ok" if v.confirmed else "reject"
        parts.append(f"{v.slug}:{flag}({v.ontology_score})")
    return ";".join(parts)
