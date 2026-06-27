"""Tiered FoodOn linking: fuzzy + semantic retrieval (+ optional LLM judge)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from foodon_mapping_io import LINKER_VERSION

AUTO_FUZZY_MIN = 0.85
AUTO_SEMANTIC_MIN = 0.78
REVIEW_SEMANTIC_MIN = 0.55


@dataclass(frozen=True)
class FoodOnLinkResult:
    fdc_id: int
    description: str
    foodon_id: str | None
    foodon_label: str | None
    match_method: str
    confidence: float
    fuzzy_score: float | None
    semantic_score: float | None
    reviewed: bool
    linker_version: str
    candidate_ids: str | None = None
    rationale: str | None = None

    def to_row(self) -> dict:
        return {
            "fdc_id": self.fdc_id,
            "description": self.description,
            "foodon_id": self.foodon_id,
            "foodon_label": self.foodon_label,
            "match_method": self.match_method,
            "confidence": self.confidence,
            "fuzzy_score": self.fuzzy_score,
            "semantic_score": self.semantic_score,
            "reviewed": self.reviewed,
            "linker_version": self.linker_version,
            "candidate_ids": self.candidate_ids,
            "rationale": self.rationale,
        }


def _candidate_json(candidates: list[dict[str, Any]], limit: int = 20) -> str:
    slim = [
        {"id": c["id"], "label": c.get("label"), "score": round(float(c.get("score", 0)), 4)}
        for c in candidates[:limit]
    ]
    return json.dumps(slim)


def link_food_to_foodon(
    fdc_id: int,
    description: str,
    *,
    foodon_index: Any,
    embed_index: Any | None = None,
    llm_judge: Any | None = None,
    fuzzy_min: float = 0.55,
    auto_fuzzy_min: float = AUTO_FUZZY_MIN,
    auto_semantic_min: float = AUTO_SEMANTIC_MIN,
) -> FoodOnLinkResult:
    """Resolve one USDA food to a FoodOn class using tiered logic."""
    desc = (description or "").strip()
    fuzzy = foodon_index.best_match(desc, min_score=fuzzy_min) if desc else None
    fuzzy_score = float(fuzzy["score"]) if fuzzy else None
    fuzzy_id = str(fuzzy["id"]) if fuzzy else None
    fuzzy_label = str(fuzzy["label"]) if fuzzy else None

    semantic_hits: list[dict[str, Any]] = []
    if embed_index is not None and desc:
        semantic_hits = embed_index.search(desc, k=20)
    sem_top = semantic_hits[0] if semantic_hits else None
    semantic_score = float(sem_top["score"]) if sem_top else None
    sem_id = str(sem_top["id"]) if sem_top else None
    sem_label = str(sem_top["label"]) if sem_top else None

    candidates = semantic_hits or ([fuzzy] if fuzzy else [])
    cand_json = _candidate_json(candidates) if candidates else None

    def _result(
        foodon_id: str | None,
        foodon_label: str | None,
        method: str,
        confidence: float,
        reviewed: bool,
        rationale: str | None = None,
    ) -> FoodOnLinkResult:
        return FoodOnLinkResult(
            fdc_id=int(fdc_id),
            description=desc,
            foodon_id=foodon_id,
            foodon_label=foodon_label,
            match_method=method,
            confidence=confidence,
            fuzzy_score=fuzzy_score,
            semantic_score=semantic_score,
            reviewed=reviewed,
            linker_version=LINKER_VERSION,
            candidate_ids=cand_json,
            rationale=rationale,
        )

    # Tier 1: high-confidence fuzzy (+ semantic agreement if available)
    if fuzzy and fuzzy_score is not None and fuzzy_score >= auto_fuzzy_min:
        agree = sem_id is None or sem_id == fuzzy_id or (semantic_score or 0) >= auto_semantic_min
        if agree:
            return _result(fuzzy_id, fuzzy_label, "auto_fuzzy", fuzzy_score, reviewed=False)

    # Tier 2: high-confidence semantic
    if sem_top and semantic_score is not None and semantic_score >= auto_semantic_min:
        return _result(sem_id, sem_label, "auto_semantic", semantic_score, reviewed=False)

    # Tier 3: optional LLM judge over semantic candidates
    if llm_judge is not None and semantic_hits:
        judged = llm_judge(desc, semantic_hits)
        if judged.get("foodon_id"):
            return _result(
                str(judged["foodon_id"]),
                judged.get("foodon_label"),
                "llm_judge",
                float(judged.get("confidence") or 0.0),
                reviewed=False,
                rationale=judged.get("rationale"),
            )
        if judged.get("abstain"):
            return _result(
                None,
                None,
                "llm_abstain",
                float(judged.get("confidence") or 0.0),
                reviewed=True,
                rationale=judged.get("rationale"),
            )

    # Tier 4: needs human review — keep best available guess
    if sem_top and semantic_score is not None and semantic_score >= REVIEW_SEMANTIC_MIN:
        return _result(sem_id, sem_label, "needs_review", semantic_score, reviewed=True)
    if fuzzy:
        return _result(fuzzy_id, fuzzy_label, "needs_review", fuzzy_score or 0.0, reviewed=True)

    return _result(None, None, "no_match", 0.0, reviewed=True)
