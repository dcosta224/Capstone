"""Pick a neighborhood example recipe near semantic query + target macros.

Used by creative_example eval/agent mode: show the LLM a real recipe that
(1) matches the dish query ingredient-wise / by title and (2) is closest in
PFC calorie shares to the high-protein target midpoint.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def _tokenize(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2}


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / len(ta | tb))


def _pfc_l1(a: dict[str, float], mid: dict[str, float]) -> float:
    return float(
        abs(float(a.get("protein", 0.0)) - float(mid.get("protein", 0.0)))
        + abs(float(a.get("carbs", 0.0)) - float(mid.get("carbs", 0.0)))
        + abs(float(a.get("fat", 0.0)) - float(mid.get("fat", 0.0)))
    )


def _ingredient_rows_for_recipe(lines_df, recipe_id: str) -> list[dict[str, Any]]:
    sub = lines_df.loc[lines_df["recipe_nlg_id"].astype(str) == str(recipe_id)]
    if sub.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in sub.itertuples(index=False):
        label = str(
            getattr(row, "fdc_description", None)
            or getattr(row, "ingredient_text", None)
            or getattr(row, "name", None)
            or "?"
        )
        grams = float(getattr(row, "gram_weight", 0.0) or 0.0)
        if grams <= 0:
            continue
        rows.append(
            {
                "label": label,
                "grams": round(grams, 1),
                "fdc_id": int(getattr(row, "fdc_id")) if getattr(row, "fdc_id", None) is not None else None,
                "role": None,
            }
        )
    return rows


def pick_example_recipe_near_targets(
    *,
    lines_df,
    recipe_ids: list[str],
    query: str,
    target_mid: dict[str, float],
    titles_by_id: dict[str, str] | None = None,
    semantic_weight: float = 0.45,
    nutrition_weight: float = 0.55,
    top_k_semantic: int = 12,
) -> dict[str, Any] | None:
    """Return the best example recipe dict, or None.

    Scoring: among the top ``top_k_semantic`` title/ingredient matches to
    ``query``, pick the one with smallest PFC L1 distance to ``target_mid``.
    Falls back to joint weighted score if fewer than 3 semantic hits.
    """
    from recipe_opt_agent.problem_loader import _batch_recipe_pfc_from_lines

    if lines_df is None or getattr(lines_df, "empty", True) or not recipe_ids:
        return None
    pfc_by_rid = _batch_recipe_pfc_from_lines(lines_df)
    if not pfc_by_rid:
        return None

    titles_by_id = titles_by_id or {}
    # Build a cheap ingredient-bag text per recipe for semantic scoring.
    label_text: dict[str, str] = {}
    for rid, sub in lines_df.groupby(lines_df["recipe_nlg_id"].astype(str)):
        descs = []
        if "fdc_description" in sub.columns:
            descs = [str(x) for x in sub["fdc_description"].dropna().tolist()[:24]]
        title = titles_by_id.get(str(rid), "")
        label_text[str(rid)] = f"{title} {' '.join(descs)}".strip()

    scored: list[dict[str, Any]] = []
    for rid in map(str, recipe_ids):
        pfc = pfc_by_rid.get(rid)
        if not pfc:
            continue
        text = label_text.get(rid) or titles_by_id.get(rid, "")
        sem = _token_jaccard(query, text)
        # Also boost pure title match
        title = titles_by_id.get(rid, "")
        if title:
            sem = max(sem, _token_jaccard(query, title))
        dist = _pfc_l1(pfc, target_mid)
        scored.append(
            {
                "recipe_nlg_id": rid,
                "title": title or rid,
                "semantic_score": sem,
                "pfc_l1_to_target": dist,
                "pfc": {
                    "protein": float(pfc["protein"]),
                    "carbs": float(pfc["carbs"]),
                    "fat": float(pfc["fat"]),
                },
            }
        )
    if not scored:
        return None

    # Prefer high semantic match, then closest nutrition among those.
    scored.sort(key=lambda r: (-r["semantic_score"], r["pfc_l1_to_target"]))
    pool = scored[: max(3, top_k_semantic)]
    # If semantic signal is weak across the board, fall back to joint score.
    best_sem = pool[0]["semantic_score"] if pool else 0.0
    if best_sem < 0.08:
        max_dist = max(r["pfc_l1_to_target"] for r in scored) or 1.0
        for r in scored:
            nutr = 1.0 - (r["pfc_l1_to_target"] / max_dist)
            r["joint_score"] = semantic_weight * r["semantic_score"] + nutrition_weight * nutr
        pick = max(scored, key=lambda r: r["joint_score"])
    else:
        pick = min(pool, key=lambda r: r["pfc_l1_to_target"])

    ingredients = _ingredient_rows_for_recipe(lines_df, pick["recipe_nlg_id"])
    if not ingredients:
        return None
    return {
        "recipe_nlg_id": pick["recipe_nlg_id"],
        "title": pick["title"],
        "ingredients": ingredients,
        "pfc": pick["pfc"],
        "semantic_score": pick["semantic_score"],
        "pfc_l1_to_target": pick["pfc_l1_to_target"],
        "selection": {
            "query": query,
            "target_mid": target_mid,
            "semantic_weight": semantic_weight,
            "nutrition_weight": nutrition_weight,
        },
    }


def attach_example_recipe_to_problem(
    problem: dict[str, Any],
    example: dict[str, Any] | None,
) -> dict[str, Any]:
    """Store example on the problem so creative draft can surface it to the LLM."""
    problem = dict(problem)
    if example:
        problem["example_recipe"] = example
        ctx = dict(problem.get("retrieval_context") or {})
        ctx["example_recipe"] = {
            "recipe_nlg_id": example.get("recipe_nlg_id"),
            "title": example.get("title"),
            "pfc": example.get("pfc"),
            "semantic_score": example.get("semantic_score"),
            "pfc_l1_to_target": example.get("pfc_l1_to_target"),
            "n_ingredients": len(example.get("ingredients") or []),
        }
        problem["retrieval_context"] = ctx
    return problem
