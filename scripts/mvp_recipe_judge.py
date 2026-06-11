"""Final recipe selection: LLM reviews each candidate; code balances fit vs portions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai_fallback import get_sync_openai_client

DEFAULT_MODEL = "gpt-4o-mini"

# Portion distortion severe enough to prefer a weaker semantic match.
PORTION_SEVERE = 1.0
AVG_PCT_SEVERE = 40.0

FINAL_SELECT_SYSTEM_PROMPT = f"""You review optimizer candidates and assess how well each satisfies the user.

Each candidate includes OFFICIAL_INGREDIENTS from recipe.recipe_nlg_features. You MUST read this
list carefully to decide what the dish actually contains. Do not infer ingredients from the title
or from words like "poultry" in a seasoning name.

For EVERY candidate return a candidate_review with:
- edible_alone: false for spice/seasoning blends, rubs, or condiment-only items (not a meal).
- query_fit: one of strong_match, weak_match, no_match
  * strong_match: clearly satisfies the user's request (e.g. user wants chicken and OFFICIAL_INGREDIENTS
    lists chicken, cooked chicken, or chicken thighs — not just "poultry seasoning").
  * weak_match: partially related but missing key aspects of the request.
  * no_match: does not satisfy the request (e.g. beef chilli when user asked for chicken).
- fit_reason: one sentence citing specific OFFICIAL_INGREDIENTS and the user's words.

Casseroles, sides, and one-pot dishes ARE valid meals when they match the request.

The system will pick the winner using your reviews plus portion metrics:
- Prefer strong_match over weak_match over no_match.
- Among same fit tier, prefer lower portion_score (less reshaping).
- A strong_match is only skipped for a weak_match when EVERY strong_match has portion_score
  >= {PORTION_SEVERE} OR avg_pct_change >= {AVG_PCT_SEVERE}% (severe distortion).

In why_this_recipe, write 2-3 sentences in plain language explaining how this recipe meets the
user's specific request — what they asked for and how this dish delivers it. Focus on the match
to their demand, not on praising the meal. Do NOT mention portion scores, semantic similarity,
optimizers, rankings, or how the system chose it.

In runner_up_notes, optionally mention other appealing options in plain language (no technical metrics).

Respond with valid JSON only."""

FINAL_SELECT_SCHEMA = {
    "name": "final_recipe_candidate_review",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "candidate_reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {"type": "integer"},
                        "edible_alone": {"type": "boolean"},
                        "query_fit": {
                            "type": "string",
                            "enum": ["strong_match", "weak_match", "no_match"],
                        },
                        "fit_reason": {"type": "string"},
                    },
                    "required": ["recipe_id", "edible_alone", "query_fit", "fit_reason"],
                    "additionalProperties": False,
                },
            },
            "why_this_recipe": {"type": "string"},
            "runner_up_notes": {"type": "string"},
        },
        "required": ["candidate_reviews", "why_this_recipe", "runner_up_notes"],
        "additionalProperties": False,
    },
}


@dataclass
class FinalPickCandidate:
    recipe_id: int
    title_clean: str
    semantic_text: str
    nlg_ingredients: str
    ingredient_summary: str
    semantic_sim: float
    portion_score: float
    avg_pct_change: float
    macro_feasible: bool = True
    used_fallback: bool = False
    already_feasible: bool = False
    stage1_rank: int = 0


@dataclass
class CandidateReview:
    recipe_id: int
    edible_alone: bool
    query_fit: str
    fit_reason: str


@dataclass
class JudgeResult:
    chosen_recipe_id: int
    rationale: str
    portion_summary: str
    runner_up_notes: str


def _pick_sort_key(c: FinalPickCandidate) -> tuple[float, float, int]:
    return (c.portion_score, c.avg_pct_change, c.stage1_rank)


def _semantic_sort_key(c: FinalPickCandidate) -> tuple[float, float, float, int]:
    return (-c.semantic_sim, c.portion_score, c.avg_pct_change, c.stage1_rank)


def _feasible_pool(candidates: list[FinalPickCandidate]) -> list[FinalPickCandidate]:
    feasible = [c for c in candidates if c.macro_feasible and not c.used_fallback]
    return feasible if feasible else list(candidates)


def _is_portion_severe(c: FinalPickCandidate) -> bool:
    return c.portion_score >= PORTION_SEVERE or c.avg_pct_change >= AVG_PCT_SEVERE


def _parse_candidate_reviews(
    raw_reviews: list[dict],
    valid_ids: set[int],
) -> dict[int, CandidateReview]:
    out: dict[int, CandidateReview] = {}
    for item in raw_reviews:
        rid = int(item.get("recipe_id", -1))
        if rid not in valid_ids:
            continue
        out[rid] = CandidateReview(
            recipe_id=rid,
            edible_alone=bool(item.get("edible_alone", True)),
            query_fit=str(item.get("query_fit", "no_match")),
            fit_reason=str(item.get("fit_reason", "")),
        )
    return out


def select_from_candidate_reviews(
    pool: list[FinalPickCandidate],
    reviews_by_id: dict[int, CandidateReview],
) -> tuple[FinalPickCandidate, str]:
    """Pick winner: query fit first, portion score second unless fit options are severely distorted."""
    eligible: list[tuple[FinalPickCandidate, CandidateReview]] = []
    for c in pool:
        rev = reviews_by_id.get(c.recipe_id)
        if not rev or not rev.edible_alone or rev.query_fit == "no_match":
            continue
        eligible.append((c, rev))

    if not eligible:
        fallback = min(pool, key=_semantic_sort_key)
        return (
            fallback,
            "No candidate passed LLM edibility/query review; fell back to best stage-1 semantic match.",
        )

    strong = [c for c, r in eligible if r.query_fit == "strong_match"]
    weak = [c for c, r in eligible if r.query_fit == "weak_match"]

    if strong:
        non_severe = [c for c in strong if not _is_portion_severe(c)]
        if non_severe:
            chosen = min(non_severe, key=_pick_sort_key)
            return (
                chosen,
                "Selected best portion fit among strong query matches.",
            )
        if weak:
            best_weak = min(weak, key=_pick_sort_key)
            best_strong = min(strong, key=_pick_sort_key)
            if best_weak.portion_score + 0.15 < best_strong.portion_score:
                return (
                    best_weak,
                    "All strong matches need severe portion reshaping; chose weaker fit with much "
                    "lower portion distortion.",
                )
        chosen = min(strong, key=_pick_sort_key)
        return (
            chosen,
            "Selected best portion fit among strong query matches (all require notable reshaping).",
        )

    chosen = min(weak, key=_pick_sort_key)
    return chosen, "No strong matches; selected best portion fit among weak matches."


def _user_facing_why(
    chosen: FinalPickCandidate,
    *,
    why_this_recipe: str = "",
    reviews_by_id: dict[int, CandidateReview] | None = None,
) -> str:
    if why_this_recipe.strip():
        return why_this_recipe.strip()
    rev = (reviews_by_id or {}).get(chosen.recipe_id)
    if rev and rev.fit_reason.strip():
        return f"This recipe meets your request: {rev.fit_reason.strip()}"
    return "This recipe best matches what you asked for among the options we considered."


def _build_judge_result(
    chosen: FinalPickCandidate,
    pool: list[FinalPickCandidate],
    *,
    why_this_recipe: str = "",
    runner_up_notes: str = "",
    reviews_by_id: dict[int, CandidateReview] | None = None,
) -> JudgeResult:
    if chosen.already_feasible:
        portion_summary = "Original portions already satisfied PFC targets; no reshaping required."
    else:
        portion_summary = (
            "Portions adjusted via convex optimizer to hit macro targets while "
            "minimizing deviation from uniform calorie scaling."
        )

    rationale = _user_facing_why(
        chosen,
        why_this_recipe=why_this_recipe,
        reviews_by_id=reviews_by_id,
    )

    return JudgeResult(
        chosen_recipe_id=chosen.recipe_id,
        rationale=rationale,
        portion_summary=portion_summary,
        runner_up_notes=runner_up_notes.strip(),
    )


def select_best_semantic_candidate(
    candidates: list[FinalPickCandidate],
    *,
    taste_text: str = "",
    note: str = "",
) -> JudgeResult:
    """Fallback when LLM judge is unavailable: stage-1 semantic sim, then portion score."""
    if not candidates:
        raise ValueError("No candidates to select from")
    pool = _feasible_pool(candidates)
    chosen = min(pool, key=_semantic_sort_key)
    result = _build_judge_result(chosen, pool)
    fallback_note = note or (
        "LLM judge unavailable (no OPENAI_API_KEY); selected by semantic similarity "
        "then portion score, without edibility or ingredient review."
    )
    return JudgeResult(
        chosen_recipe_id=result.chosen_recipe_id,
        rationale=result.rationale,
        portion_summary=result.portion_summary,
        runner_up_notes=fallback_note,
    )


def build_final_select_prompt(taste_text: str, candidates: list[FinalPickCandidate]) -> str:
    lines = [
        f"User taste preferences: {taste_text}",
        "",
        "Review EVERY candidate. Use OFFICIAL_INGREDIENTS (from recipe.recipe_nlg_features) as the",
        "source of truth for what each recipe contains.",
        "",
        "Candidates:",
    ]
    for i, c in enumerate(candidates, start=1):
        lines.append(f"\n--- #{i} Recipe {c.recipe_id}: {c.title_clean} ---")
        lines.append(
            "OFFICIAL_INGREDIENTS (recipe.recipe_nlg_features): "
            f"{c.nlg_ingredients or '(not available)'}"
        )
        lines.append(f"portion_score: {c.portion_score:.4f} (lower = less reshaping; severe if >= {PORTION_SEVERE})")
        lines.append(
            f"avg_pct_change: {c.avg_pct_change:.1f}% (severe if >= {AVG_PCT_SEVERE}%)"
        )
        lines.append(f"semantic_sim: {c.semantic_sim:.3f} (stage-1 embedding similarity)")
        lines.append(f"stage1_rank: {c.stage1_rank}")
        lines.append(f"already_feasible: {c.already_feasible}")
        if c.ingredient_summary:
            lines.append(f"Resolved ingredients (USDA-linked): {c.ingredient_summary}")
        if c.semantic_text:
            lines.append(f"Description: {c.semantic_text[:400]}")
    return "\n".join(lines)


def llm_select_final_candidate(
    taste_text: str,
    candidates: list[FinalPickCandidate],
    *,
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    if not candidates:
        raise ValueError("No candidates to select from")

    pool = _feasible_pool(candidates)
    valid_ids = {c.recipe_id for c in pool}

    client = get_sync_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FINAL_SELECT_SYSTEM_PROMPT},
            {"role": "user", "content": build_final_select_prompt(taste_text, pool)},
        ],
        response_format={"type": "json_schema", "json_schema": FINAL_SELECT_SCHEMA},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content)

    reviews_by_id = _parse_candidate_reviews(
        parsed.get("candidate_reviews", []),
        valid_ids,
    )
    chosen, _pick_note = select_from_candidate_reviews(pool, reviews_by_id)

    return _build_judge_result(
        chosen,
        pool,
        why_this_recipe=str(parsed.get("why_this_recipe", "")),
        runner_up_notes=str(parsed.get("runner_up_notes", "")),
        reviews_by_id=reviews_by_id,
    )


def select_final_candidate(
    taste_text: str,
    candidates: list[FinalPickCandidate],
) -> JudgeResult:
    """LLM per-candidate review, then fit-first selection with portion tie-break."""
    if not candidates:
        raise ValueError("No candidates to select from")

    if os.environ.get("OPENAI_API_KEY", "").strip():
        return llm_select_final_candidate(taste_text, candidates)
    return select_best_semantic_candidate(candidates, taste_text=taste_text)


# Backwards-compatible alias for tests
select_best_portion_candidate = select_best_semantic_candidate
