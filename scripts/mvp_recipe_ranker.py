"""Stage-1 recipe ranker: semantic similarity + PFC nutrient fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mvp_nutrient_fit import (
    clamp_fraction_bounds,
    kcal_target_midpoint,
    nutrient_fit_to_score,
    nutrient_range_fit_from_totals,
    pfc_in_range,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PFC_FIT_TOL = 1e-6


@dataclass
class RankedRecipe:
    recipe_id: int
    recipe_name: str
    semantic_sim: float
    semantic_dist: float
    semantic_score: float
    nutrient_fit: float
    nutrient_score: float
    combined_score: float
    rank: int
    pfc_in_range: bool
    kcal_target: float
    recipe_kcal: float


def cosine_similarity(query_emb: np.ndarray, recipe_embs: np.ndarray) -> np.ndarray:
    q = np.asarray(query_emb, dtype=float).ravel()
    E = np.asarray(recipe_embs, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return np.zeros(len(E))
    En = np.linalg.norm(E, axis=1)
    En = np.where(En == 0, 1.0, En)
    return (E @ q) / (En * qn)


def rank_recipes(
    recipe_ids: list[int],
    recipe_names: list[str],
    recipe_embs: np.ndarray,
    query_emb: np.ndarray,
    nutrient_rows: list[dict[str, Any]],
    *,
    kcal_min: float,
    kcal_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    protein_frac_min: float,
    protein_frac_max: float,
    w_semantic: float = 0.5,
    w_nutrient: float = 0.5,
) -> list[RankedRecipe]:
    f0, f1, c0, c1, p0, p1 = clamp_fraction_bounds(
        fat_frac_min,
        fat_frac_max,
        carb_frac_min,
        carb_frac_max,
        protein_frac_min,
        protein_frac_max,
    )
    kcal_target = kcal_target_midpoint(kcal_min, kcal_max)

    sims = cosine_similarity(query_emb, recipe_embs)
    sem_dists = 1.0 - sims

    nutrient_fits = []
    pfc_flags = []
    recipe_kcals = []
    for row in nutrient_rows:
        protein = float(row["protein_g"])
        fat = float(row["total_lipid_fat_g"])
        carbs = float(row["carbohydrate_by_difference_g"])
        kcal = float(row["energy_kcal"])
        recipe_kcals.append(kcal)
        fit = nutrient_range_fit_from_totals(
            protein_g=protein,
            fat_g=fat,
            carbs_g=carbs,
            energy_kcal=kcal,
            fat_frac_min=f0,
            fat_frac_max=f1,
            carb_frac_min=c0,
            carb_frac_max=c1,
            protein_frac_min=p0,
            protein_frac_max=p1,
        )
        nutrient_fits.append(fit)
        pfc_flags.append(
            pfc_in_range(
                protein,
                fat,
                carbs,
                kcal,
                f0,
                f1,
                c0,
                c1,
                p0,
                p1,
                tol=PFC_FIT_TOL,
            )
        )

    nutrient_fits_arr = np.array(nutrient_fits, dtype=float)
    w_sum = w_semantic + w_nutrient
    if w_sum <= 0:
        w_semantic, w_nutrient, w_sum = 0.5, 0.5, 1.0
    w_semantic /= w_sum
    w_nutrient /= w_sum

    semantic_scores = np.clip(sims, 0.0, 1.0)
    nutrient_scores = np.array(
        [
            nutrient_fit_to_score(
                float(nutrient_fits_arr[i]),
                in_range=bool(pfc_flags[i]),
                tol=PFC_FIT_TOL,
            )
            for i in range(len(nutrient_fits_arr))
        ],
        dtype=float,
    )
    combined = 100.0 * (w_semantic * semantic_scores + w_nutrient * nutrient_scores)

    order = np.argsort(-combined)
    results: list[RankedRecipe] = []
    for rank_idx, i in enumerate(order):
        results.append(
            RankedRecipe(
                recipe_id=int(recipe_ids[i]),
                recipe_name=str(recipe_names[i]),
                semantic_sim=float(sims[i]),
                semantic_dist=float(sem_dists[i]),
                semantic_score=float(semantic_scores[i]),
                nutrient_fit=float(nutrient_fits_arr[i]),
                nutrient_score=float(nutrient_scores[i]),
                combined_score=float(combined[i]),
                rank=rank_idx + 1,
                pfc_in_range=bool(pfc_flags[i]),
                kcal_target=kcal_target,
                recipe_kcal=float(recipe_kcals[i]),
            )
        )
    return results
