"""MVP recipe recommendation pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from db import connect
from mvp_corpus_cache import get_cached_food_nutrients, get_cached_ingredients, get_mvp_corpus
from mvp_data import build_recipe_macro_inputs, parse_nlg_ingredients
from mvp_log import finish_run, log_stage, start_run
from mvp_recipe_judge import FinalPickCandidate, select_final_candidate
from mvp_recipe_ranker import EMBEDDING_MODEL, rank_recipes
from mvp_nutrient_fit import kcal_target_midpoint
from recipe_macro_optimizer import (
    IngredientMeta,
    OptimizerConfig,
    RecipeMacroOptimizer,
    derive_macro_bounds_from_fractions,
    format_serving_display,
    macros_to_dict,
)

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def encode_query(text: str) -> np.ndarray:
    model = get_embedding_model()
    emb = model.encode([text], normalize_embeddings=True)
    return np.asarray(emb[0], dtype=np.float32)


@dataclass
class UserQuery:
    taste_text: str
    kcal_min: float
    kcal_max: float
    fat_frac_min: float
    fat_frac_max: float
    carb_frac_min: float
    carb_frac_max: float
    protein_frac_min: float
    protein_frac_max: float
    w_semantic: float = 0.5
    w_nutrient: float = 0.5
    top_k: int = 10


@dataclass
class PipelineEvent:
    stage: str
    seq: int
    payload: dict[str, Any]


def _emit(
    events: list[PipelineEvent],
    run_id: str | None,
    stage: str,
    seq: int,
    payload: dict[str, Any],
    on_event: Callable[[PipelineEvent], None] | None,
) -> PipelineEvent:
    ev = PipelineEvent(stage=stage, seq=seq, payload=payload)
    events.append(ev)
    if on_event:
        on_event(ev)
    if run_id:
        log_stage(run_id, stage, seq, payload)
    return ev


def _ranked_to_dict(r) -> dict[str, Any]:
    return {
        "recipe_id": r.recipe_id,
        "recipe_name": r.recipe_name,
        "semantic_sim": r.semantic_sim,
        "semantic_score": r.semantic_score,
        "nutrient_fit": r.nutrient_fit,
        "nutrient_score": r.nutrient_score,
        "combined_score": r.combined_score,
        "rank": r.rank,
        "pfc_in_range": r.pfc_in_range,
        "kcal_target": r.kcal_target,
        "recipe_kcal": r.recipe_kcal,
    }


def optimize_recipe(
    recipe_id: int,
    query: UserQuery,
    *,
    corpus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corpus = corpus or get_mvp_corpus()
    ingredients = get_cached_ingredients(corpus, recipe_id)
    fdc_ids = [int(x) for x in ingredients["fdc_id"].dropna().astype(int).tolist()]
    food_nutrients = get_cached_food_nutrients(corpus, fdc_ids)

    x0, M = build_recipe_macro_inputs(ingredients, food_nutrients)
    bounds = derive_macro_bounds_from_fractions(
        query.kcal_min,
        query.kcal_max,
        query.fat_frac_min,
        query.fat_frac_max,
        query.carb_frac_min,
        query.carb_frac_max,
        query.protein_frac_min,
        query.protein_frac_max,
    )
    opt = RecipeMacroOptimizer()
    result = opt.optimize(x0, M, OptimizerConfig(macro_bounds=bounds))

    meta = [
        IngredientMeta(
            ingredient_idx=int(row.ingredient_idx),
            ingredient=str(row.ingredient),
            fdc_id=int(row.fdc_id) if row.fdc_id is not None else None,
            quantity=float(row.quantity) if row.quantity is not None else None,
            unit=str(row.unit) if row.unit else None,
            portion_label=str(row.portion_label) if row.portion_label else None,
            fdc_description=str(row.fdc_description) if row.fdc_description else None,
        )
        for row in ingredients.itertuples(index=False)
    ]
    servings = format_serving_display(result, x0, meta)
    feat = corpus.get("features", {}).get(recipe_id, {})
    recipe_name = feat.get("title_clean") or f"Recipe {recipe_id}"
    return {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "portion_score": result.portion_score,
        "avg_pct_change": result.avg_pct_change,
        "max_pct_change": result.max_pct_change,
        "optimizer_status": result.status,
        "sca_iters": result.sca_iters,
        "converged": result.converged,
        "macro_feasible": result.macro_feasible,
        "feasibility_message": result.feasibility_message,
        "used_fallback": result.used_fallback,
        "already_feasible": result.already_feasible,
        "kcal_target": result.kcal_target,
        "constraint_violations": result.constraint_violations,
        "macros_before": macros_to_dict(result.macros_before),
        "macros_after": macros_to_dict(result.macros_after),
        "macro_slack": result.macro_slack.tolist(),
        "ingredients": servings,
        "max_r": float(np.max(result.r)),
    }


def run_pipeline(
    query: UserQuery,
    *,
    on_event: Callable[[PipelineEvent], None] | None = None,
    log_to_db: bool = True,
    corpus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events: list[PipelineEvent] = []
    seq = 0
    run_id = None
    params = asdict(query)

    if log_to_db:
        try:
            run_id = start_run(query.taste_text, params)
        except Exception:
            run_id = None

    try:
        kcal_target = kcal_target_midpoint(query.kcal_min, query.kcal_max)

        seq += 1
        _emit(
            events,
            run_id,
            "embed_query",
            seq,
            {"message": "Encoding taste preference…", "model": EMBEDDING_MODEL, "status": "running"},
            on_event,
        )
        query_emb = encode_query(query.taste_text)
        _emit(
            events,
            run_id,
            "embed_query",
            seq,
            {"message": "Taste embedding complete", "model": EMBEDDING_MODEL, "status": "done"},
            on_event,
        )

        if corpus is None:
            corpus = get_mvp_corpus()

        seq += 1
        _emit(
            events,
            run_id,
            "stage1_rank",
            seq,
            {"message": "Ranking recipes by semantic + PFC fit…", "status": "running", "kcal_target": kcal_target},
            on_event,
        )
        ranked = rank_recipes(
            corpus["recipe_ids"],
            corpus["recipe_names"],
            corpus["embeddings"],
            query_emb,
            corpus["nutrient_rows"],
            kcal_min=query.kcal_min,
            kcal_max=query.kcal_max,
            fat_frac_min=query.fat_frac_min,
            fat_frac_max=query.fat_frac_max,
            carb_frac_min=query.carb_frac_min,
            carb_frac_max=query.carb_frac_max,
            protein_frac_min=query.protein_frac_min,
            protein_frac_max=query.protein_frac_max,
            w_semantic=query.w_semantic,
            w_nutrient=query.w_nutrient,
        )
        _emit(
            events,
            run_id,
            "stage1_rank",
            seq,
            {
                "message": "Ranking complete",
                "status": "done",
                "n_recipes": len(ranked),
                "kcal_target": kcal_target,
                "top_20": [_ranked_to_dict(r) for r in ranked[:20]],
            },
            on_event,
        )

        top_ids = [r.recipe_id for r in ranked[: query.top_k]]
        seq += 1
        _emit(
            events,
            run_id,
            "optimize",
            seq,
            {
                "message": f"Optimizing top {len(top_ids)} recipes…",
                "status": "running",
                "kcal_target": kcal_target,
                "total": len(top_ids),
                "completed": 0,
                "candidates": [],
            },
            on_event,
        )
        optimized = []
        for idx, rid in enumerate(top_ids):
            candidate = optimize_recipe(rid, query, corpus=corpus)
            optimized.append(candidate)
            _emit(
                events,
                run_id,
                "optimize_progress",
                seq,
                {
                    "index": idx + 1,
                    "total": len(top_ids),
                    "candidate": candidate,
                    "status": "running",
                },
                on_event,
            )
        n_infeasible = sum(1 for o in optimized if not o.get("macro_feasible", True))
        n_fallback = sum(1 for o in optimized if o.get("used_fallback"))
        n_already = sum(1 for o in optimized if o.get("already_feasible"))
        _emit(
            events,
            run_id,
            "optimize",
            seq,
            {
                "message": "Optimization complete",
                "status": "done",
                "kcal_target": kcal_target,
                "candidates": optimized,
                "n_infeasible": n_infeasible,
                "n_fallback": n_fallback,
                "n_already_feasible": n_already,
                "completed": len(optimized),
                "total": len(top_ids),
                "infeasible_note": (
                    f"{n_infeasible} of {len(optimized)} recipes cannot reach PFC targets "
                    f"at {kcal_target:.0f} kcal."
                    if n_infeasible
                    else None
                ),
            },
            on_event,
        )

        seq += 1
        _emit(
            events,
            run_id,
            "judge",
            seq,
            {"message": "Reranking for taste fit…", "status": "running"},
            on_event,
        )

        rank_by_id = {r.recipe_id: r.rank for r in ranked}
        sim_by_id = {r.recipe_id: r.semantic_sim for r in ranked}
        pick_candidates = []
        for opt in optimized:
            rid = opt["recipe_id"]
            feat = corpus["features"].get(rid, {})
            semantic_text = feat.get("semantic_text", "")
            nlg_ingredients = feat.get("nlg_ingredients") or parse_nlg_ingredients(semantic_text)
            ing_names = ", ".join(i["ingredient"] for i in opt["ingredients"])
            pick_candidates.append(
                FinalPickCandidate(
                    recipe_id=rid,
                    title_clean=feat.get("title_clean", f"Recipe {rid}"),
                    semantic_text=semantic_text,
                    nlg_ingredients=nlg_ingredients,
                    ingredient_summary=ing_names,
                    semantic_sim=float(sim_by_id.get(rid, 0.0)),
                    portion_score=float(opt["portion_score"]),
                    avg_pct_change=float(opt.get("avg_pct_change", 0.0)),
                    macro_feasible=bool(opt.get("macro_feasible", True)),
                    used_fallback=bool(opt.get("used_fallback", False)),
                    already_feasible=bool(opt.get("already_feasible", False)),
                    stage1_rank=int(rank_by_id.get(rid, 999)),
                )
            )

        judge_result = select_final_candidate(query.taste_text, pick_candidates)

        chosen_id = judge_result.chosen_recipe_id
        chosen_opt = next(o for o in optimized if o["recipe_id"] == chosen_id)
        chosen_feat = corpus["features"].get(chosen_id, {})
        chosen_name = chosen_feat.get("title_clean") or f"Recipe {chosen_id}"

        _emit(
            events,
            run_id,
            "judge",
            seq,
            {
                "message": f"Selected {chosen_name}",
                "status": "done",
                "chosen_recipe_id": chosen_id,
                "rationale_preview": judge_result.rationale[:200],
            },
            on_event,
        )

        seq += 1
        _emit(
            events,
            run_id,
            "finalize",
            seq,
            {"message": "Assembling final recommendation…", "status": "running"},
            on_event,
        )

        seq += 1
        if chosen_opt.get("already_feasible"):
            opt_note = chosen_opt.get("feasibility_message")
        elif chosen_opt.get("used_fallback"):
            opt_note = chosen_opt.get("feasibility_message")
        else:
            opt_note = None

        final_payload = {
            "run_id": run_id,
            "chosen_recipe_id": chosen_id,
            "recipe_name": chosen_feat.get("title_clean", f"Recipe {chosen_id}"),
            "semantic_text": chosen_feat.get("semantic_text", ""),
            "ingredients": chosen_opt["ingredients"],
            "macros_before": chosen_opt["macros_before"],
            "macros_after": chosen_opt["macros_after"],
            "portion_score": chosen_opt["portion_score"],
            "avg_pct_change": chosen_opt.get("avg_pct_change", 0.0),
            "max_pct_change": chosen_opt.get("max_pct_change", 0.0),
            "kcal_target": kcal_target,
            "macro_feasible": chosen_opt.get("macro_feasible", True),
            "already_feasible": chosen_opt.get("already_feasible", False),
            "feasibility_message": chosen_opt.get("feasibility_message"),
            "used_fallback": chosen_opt.get("used_fallback", False),
            "optimization_note": opt_note,
            "judge": {
                "rationale": judge_result.rationale,
                "portion_summary": judge_result.portion_summary,
                "runner_up_notes": judge_result.runner_up_notes,
            },
            "stage1_top_20": [_ranked_to_dict(r) for r in ranked[:20]],
            "optimizer_candidates": optimized,
        }
        _emit(
            events,
            run_id,
            "format_result",
            seq,
            {**final_payload, "status": "done"},
            on_event,
        )

        seq += 1
        _emit(events, run_id, "done", seq, {"status": "done"}, on_event)

        if run_id:
            finish_run(run_id, status="done", chosen_recipe_id=chosen_id)

        return final_payload

    except Exception as exc:
        if run_id:
            finish_run(run_id, status="error", error_message=str(exc))
        raise


def run_pipeline_events(
    query: UserQuery,
    **kwargs: Any,
) -> Iterator[PipelineEvent]:
    collected: list[PipelineEvent] = []

    def _collector(ev: PipelineEvent) -> None:
        collected.append(ev)

    run_pipeline(query, on_event=_collector, **kwargs)
    yield from collected
