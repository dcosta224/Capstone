"""Contextualized score bands + per-ingredient display for the Web UI.

Ratio loss = ``ratio_surrogate`` from the *chosen* recipe's ``term_losses``
(not a raw pasta∶egg ratio, and not a silent 0 when samples are missing).

Nutrient loss = PFC box slack recomputed from the chosen recipe's ``pfc_after``
vs the run's macro targets (not an unchecked telemetry zero).

Missing / unavailable signals render as ``None`` with band ``unknown`` so the
UI shows blanks instead of a false perfect score.
"""

from __future__ import annotations

from typing import Any

import numpy as np


RATIO_LOSS_BANDS = {
    "good_max": 0.015,
    "warn_max": 0.040,
    "prior_p50": 0.012,
    "prior_p75": 0.028,
    "prior_p90": 0.055,
}

NUTRIENT_LOSS_BANDS = {
    # Solver slack noise below 0.05% renders as "0.000" — treat it as green.
    "good_max": 0.0005,
    "warn_max": 0.025,
    "prior_p50": 0.0,
    "prior_p75": 0.02,
    "prior_p90": 0.06,
}

SHARE_LOSS_BANDS = {
    "good_max": 0.05,
    "warn_max": 0.15,
}


def band_for_loss(value: float | None, *, good_max: float, warn_max: float) -> str:
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v <= good_max:
        return "good"
    if v <= warn_max:
        return "warn"
    return "bad"


def band_for_holistic_0_10(score: float | None) -> str:
    if score is None:
        return "unknown"
    try:
        v = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if v >= 8.0:
        return "good"
    if v >= 5.0:
        return "warn"
    return "bad"


def empty_display_scores() -> dict[str, Any]:
    """Blank score cards before any agent iteration."""
    return {
        "ready": False,
        "iteration": None,
        "ratio_loss": {
            "value": None,
            "band": "unknown",
            "label": "Ratio loss",
            "direction": "lower_better",
            "unit": "surrogate",
            "explanation": "Neighborhood pasta∶egg ratio surrogate (lower better).",
            "thresholds": dict(RATIO_LOSS_BANDS),
            "source": None,
        },
        "nutrient_loss": {
            "value": None,
            "band": "unknown",
            "label": "Nutrient loss",
            "direction": "lower_better",
            "unit": "pfc_slack",
            "explanation": "L1 calorie-fraction distance outside the protein/carb/fat box.",
            "thresholds": dict(NUTRIENT_LOSS_BANDS),
            "source": None,
        },
        "holistic_0_10": {
            "value": None,
            "band": "unknown",
            "label": "Holistic",
            "direction": "higher_better",
            "unit": "0_10",
            "source": None,
            "explanation": "LLM judge score 0–10 when available.",
        },
        "ingredients": [],
        "score_history": [],
        "status": None,
        "title": None,
    }


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _macro_box_from_payload(payload: dict[str, Any]) -> dict[str, float] | None:
    cfg = payload.get("config") or {}
    tel = payload.get("run_telemetry") or {}
    box = payload.get("macro_targets") or tel.get("macro_targets") or {}
    if not box and cfg:
        box = {
            "protein_min": cfg.get("protein_min"),
            "protein_max": cfg.get("protein_max"),
            "carb_min": cfg.get("carb_min"),
            "carb_max": cfg.get("carb_max"),
            "fat_min": cfg.get("fat_min"),
            "fat_max": cfg.get("fat_max"),
        }
    keys = ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max")
    if not all(box.get(k) is not None for k in keys):
        return None
    return {k: float(box[k]) for k in keys}


def _pfc_box_slack(pfc: dict[str, Any] | None, box: dict[str, float] | None) -> float | None:
    if not pfc or not box:
        return None
    try:
        p = float(pfc["protein"])
        c = float(pfc["carbs"] if "carbs" in pfc else pfc.get("carb"))
        f = float(pfc["fat"])
    except (KeyError, TypeError, ValueError):
        return None

    def _axis(v: float, lo: float, hi: float) -> float:
        return max(lo - v, 0.0) + max(v - hi, 0.0)

    return (
        _axis(p, box["protein_min"], box["protein_max"])
        + _axis(c, box["carb_min"], box["carb_max"])
        + _axis(f, box["fat_min"], box["fat_max"])
    )


def resolve_chosen_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Pick opt / problem / ingredients for the recipe that scores are about."""
    payload = payload or {}
    chosen = payload.get("chosen") or {}
    entry = chosen.get("entry") if isinstance(chosen.get("entry"), dict) else None
    # Creative scored finalist nests pool entry under metrics.raw.entry
    nested = None
    if entry and isinstance(entry.get("metrics"), dict):
        nested = (entry.get("metrics") or {}).get("raw", {}).get("entry")
    pool_entry = nested if isinstance(nested, dict) else entry

    opt = None
    if isinstance(pool_entry, dict):
        opt = pool_entry.get("opt")
    if not opt and isinstance(chosen, dict):
        opt = chosen.get("opt")
    if not opt:
        opt = payload.get("opt")

    problem = payload.get("problem") or {}
    if isinstance(pool_entry, dict) and pool_entry.get("next_problem"):
        problem = pool_entry["next_problem"]

    ings = None
    for cand in (
        chosen.get("ingredients") if isinstance(chosen, dict) else None,
        (pool_entry or {}).get("ingredients") if isinstance(pool_entry, dict) else None,
        (problem.get("chosen_recipe") or {}).get("ingredients"),
        payload.get("ingredients"),
    ):
        if isinstance(cand, list) and cand:
            ings = cand
            break

    foodon = (
        payload.get("foodon_basis_report")
        or (chosen.get("foodon_basis_report") if isinstance(chosen, dict) else None)
        or (pool_entry or {}).get("foodon_basis_report")
        or (problem.get("foodon_basis_report") if isinstance(problem, dict) else None)
        or ((problem.get("chosen_recipe") or {}).get("foodon_basis_report") if isinstance(problem, dict) else None)
    )
    return {
        "opt": opt or {},
        "problem": problem or {},
        "ingredients": ings or [],
        "foodon_basis_report": foodon if isinstance(foodon, dict) else {},
        "pool_entry": pool_entry,
        "pfc_after": (opt or {}).get("pfc_after") if isinstance(opt, dict) else None,
    }


def _ratio_loss_from_term_losses(
    tl: dict[str, Any] | None,
    *,
    ratio_samples_n: int = 0,
) -> tuple[float | None, str | None]:
    """Return (value, source). Never invent 0 from missing keys / empty samples."""
    if not isinstance(tl, dict) or not tl:
        return None, None
    if "ratio_surrogate" in tl and tl["ratio_surrogate"] is not None:
        v = _as_float(tl["ratio_surrogate"])
        # Empty ratio_samples force the optimizer to report 0 — treat as missing.
        if ratio_samples_n <= 0 and v == 0.0:
            return None, None
        return v, "ratio_surrogate"
    for key in ("ratio_loss", "ratio"):
        if key in tl and tl[key] is not None:
            v = _as_float(tl[key])
            return v, key
    return None, None


def extract_ratio_and_nutrient(
    payload: dict[str, Any],
) -> tuple[float | None, str | None, float | None, str | None]:
    """Authoritative ratio + nutrient from the chosen recipe context."""
    ctx = resolve_chosen_context(payload)
    opt = ctx["opt"]
    tl = opt.get("term_losses") if isinstance(opt, dict) else None
    ratio_n = len((ctx.get("problem") or {}).get("ratio_samples") or [])
    ratio, ratio_src = _ratio_loss_from_term_losses(tl, ratio_samples_n=ratio_n)
    # Fallback: sum of per-basis share losses (still a fidelity / ratio-family signal)
    if ratio is None and isinstance(tl, dict):
        share_parts = []
        for k, v in tl.items():
            ks = str(k)
            if ks.endswith("__share") or ks in {"ratio_surrogate", "ratio_value", "ratio_loss", "ratio"}:
                continue
            fv = _as_float(v)
            if fv is not None:
                share_parts.append(fv)
        if share_parts:
            ratio, ratio_src = float(sum(share_parts)), "share_losses_sum"

    # Prefer recomputed slack from chosen pfc + box over stale telemetry.
    box = _macro_box_from_payload(payload)
    pfc = ctx.get("pfc_after") or (opt.get("pfc_after") if isinstance(opt, dict) else None)
    nutrient = _pfc_box_slack(pfc, box)
    nutrient_src = "pfc_box_slack" if nutrient is not None else None
    if nutrient is None:
        tel = payload.get("run_telemetry") or {}
        nutrient = _as_float(tel.get("final_nutrient_slack"))
        if nutrient is not None:
            nutrient_src = "telemetry_final_nutrient_slack"

    # Telemetry fallback only when marked as coming from ratio_surrogate (not share sums).
    if ratio is None:
        tel = payload.get("run_telemetry") or {}
        if tel.get("final_ratio_source") in {"ratio_surrogate", "ratio_loss", "ratio"}:
            tr = _as_float(tel.get("final_ratio_term"))
            if tr is not None:
                ratio, ratio_src = tr, f"telemetry_{tel.get('final_ratio_source')}"

    return ratio, ratio_src, nutrient, nutrient_src


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def _iqr_stats(samples: list[float] | None) -> dict[str, float] | None:
    if not samples:
        return None
    vals = sorted(float(x) for x in samples if x is not None)
    if not vals:
        return None
    return {
        "min": float(vals[0]),
        "q1": _percentile(vals, 0.25),
        "median": _percentile(vals, 0.50),
        "q3": _percentile(vals, 0.75),
        "max": float(vals[-1]),
        "n": float(len(vals)),
    }


def _calories_for_column(M: np.ndarray, i: int, grams: float) -> float | None:
    if M.ndim != 2 or i < 0 or i >= M.shape[1]:
        return None
    col = M[:, i]
    # Prefer Atwater from rows 0–2 (matches LP); fall back to energy row.
    if col.size >= 3:
        atw = float(grams) * (4.0 * float(col[0]) + 9.0 * float(col[1]) + 4.0 * float(col[2]))
        if atw > 0:
            return atw
    if col.size >= 4 and float(col[3]) > 0:
        return float(grams) * float(col[3])
    return atw if col.size >= 3 else None


def build_ingredient_display_rows(
    *,
    ingredients: list[dict[str, Any]],
    problem: dict[str, Any],
    opt: dict[str, Any],
    foodon_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Per-ingredient FoodOn, basis, levels, loss band, kcal, IQR sparkline stats."""
    foodon_report = foodon_report or {}
    fo_rows = list(foodon_report.get("ingredients") or [])
    basis_list = list(problem.get("ingredient_basis") or [])
    samples_map = problem.get("basis_samples") or {}
    M = np.asarray(problem.get("M") or [], dtype=float)
    x_opt = np.asarray(opt.get("x_opt") or problem.get("x_opt") or problem.get("x0") or [], dtype=float)
    total_mass = float(problem.get("total_mass") or (float(x_opt.sum()) if x_opt.size else 0.0) or 0.0)
    tl = opt.get("term_losses") or {}

    # Recipe share per basis node
    share_by_basis: dict[str, float] = {}
    if total_mass > 0 and x_opt.size and basis_list:
        n = min(len(basis_list), x_opt.size)
        for i in range(n):
            nid = basis_list[i]
            if not nid:
                continue
            share_by_basis[str(nid)] = share_by_basis.get(str(nid), 0.0) + float(x_opt[i]) / total_mass

    out: list[dict[str, Any]] = []
    for i, row in enumerate(ingredients):
        fo = fo_rows[i] if i < len(fo_rows) else {}
        label = row.get("label") or row.get("name") or fo.get("label") or "?"
        grams = _as_float(row.get("grams") if row.get("grams") is not None else row.get("gram_weight"))
        if grams is None and i < x_opt.size:
            grams = float(x_opt[i])
        basis = fo.get("basis_node_id") or (basis_list[i] if i < len(basis_list) else None)
        basis = str(basis) if basis else None
        leaf = fo.get("foodon_leaf_id") or row.get("foodon_id")
        leaf = str(leaf) if leaf else None
        levels = fo.get("aggregation_levels")
        if levels is not None:
            try:
                levels = int(levels)
            except (TypeError, ValueError):
                levels = None

        loss_contrib = _as_float(tl.get(basis)) if basis else None
        samples = None
        n_hits = 0
        if basis and basis in samples_map:
            try:
                samples = [float(x) for x in samples_map[basis]]
                n_hits = len(samples)
            except Exception:
                samples = None
        # If optimizer skipped this node (not in marginal_nodes / empty samples),
        # still compute an on-the-fly share MAD so the UI never blanks a mapped ingredient.
        recipe_share = share_by_basis.get(basis) if basis else None
        if recipe_share is None and grams is not None and total_mass > 0:
            recipe_share = float(grams) / total_mass
        if loss_contrib is None and recipe_share is not None and samples:
            # Mean absolute deviation vs neighborhood shares (same as empirical CDF L1).
            arr = np.asarray(samples, dtype=float)
            loss_contrib = float(np.mean(np.abs(arr - float(recipe_share))))
        elif loss_contrib is None and basis and str(basis) not in {"ood_lean_protein", "None"}:
            # Mapped but no neighborhood hits yet — mark unknown, not zero.
            loss_contrib = None

        loss_band = band_for_loss(
            loss_contrib,
            good_max=SHARE_LOSS_BANDS["good_max"],
            warn_max=SHARE_LOSS_BANDS["warn_max"],
        )
        if loss_contrib is None and n_hits < 5:
            loss_band = "unknown"

        iqr = _iqr_stats(samples)

        kcal = None
        if grams is not None and M.ndim == 2 and i < M.shape[1]:
            kcal = _calories_for_column(M, i, grams)

        out.append(
            {
                "index": i,
                "label": label,
                "grams": grams,
                "fdc_id": row.get("fdc_id"),
                "foodon_leaf_id": leaf,
                "foodon_leaf_label": fo.get("foodon_leaf_label") or leaf,
                "basis_node_id": basis,
                "basis_node_label": fo.get("basis_node_label") or basis,
                "aggregation_levels": levels,
                "loss_contribution": loss_contrib,
                "loss_band": loss_band,
                "loss_label": "ratio loss",
                "basis_n_hits": n_hits,
                "calories": kcal,
                "recipe_share": recipe_share,
                "share_iqr": iqr,
            }
        )
    return out


def _extract_holistic_0_10(payload: dict[str, Any]) -> tuple[float | None, str]:
    feval = payload.get("final_evaluation") or {}
    if isinstance(feval, dict) and feval.get("overall_score_0_10") is not None:
        v = _as_float(feval["overall_score_0_10"])
        if v is not None:
            return max(0.0, min(10.0, v)), "gpt-4o_final_evaluator"

    judge = payload.get("judge_result") or (payload.get("chosen") or {}).get("judge") or {}
    if isinstance(judge, dict):
        for key in ("holistic_score_0_10", "holistic_0_10", "score_0_10"):
            if judge.get(key) is not None:
                v = _as_float(judge[key])
                if v is not None:
                    return max(0.0, min(10.0, v)), "llm_judge"
        scores = judge.get("scores_0_10") or {}
        wid = judge.get("winner_id")
        if wid and isinstance(scores, dict) and scores.get(wid) is not None:
            v = _as_float(scores[wid])
            if v is not None:
                return max(0.0, min(10.0, v)), "llm_judge"

    tel = payload.get("run_telemetry") or {}
    if tel.get("final_holistic") is not None:
        v = _as_float(tel["final_holistic"])
        if v is not None:
            return max(0.0, min(10.0, round(v * 10.0, 1))), "intent_overlap"

    scored = payload.get("scored_finalists") or []
    winner_id = None
    chosen = payload.get("chosen") or {}
    if isinstance(chosen.get("entry"), dict):
        winner_id = chosen["entry"].get("candidate_id")
    if not winner_id and scored:
        winner_id = scored[0].get("candidate_id")
    for s in scored:
        if s.get("candidate_id") == winner_id and s.get("composite") is not None:
            v = _as_float(s["composite"])
            if v is not None:
                return max(0.0, min(10.0, round(v * 10.0, 1))), "composite"
    return None, "missing"


def build_display_scores(payload: dict[str, Any] | None, *, ready: bool = True) -> dict[str, Any]:
    """Compact payload for live + final score UI."""
    if not payload:
        return empty_display_scores()
    if not ready and not payload.get("opt") and not payload.get("chosen"):
        return empty_display_scores()

    ratio, ratio_src, nutrient, nutrient_src = extract_ratio_and_nutrient(payload)
    holistic, holistic_source = _extract_holistic_0_10(payload)
    ctx = resolve_chosen_context(payload)
    ingredients = build_ingredient_display_rows(
        ingredients=ctx["ingredients"],
        problem=ctx["problem"],
        opt=ctx["opt"],
        foodon_report=ctx["foodon_basis_report"],
    )

    # Do not color a missing signal as "good".
    ratio_band = band_for_loss(
        ratio, good_max=RATIO_LOSS_BANDS["good_max"], warn_max=RATIO_LOSS_BANDS["warn_max"]
    )
    nutrient_band = band_for_loss(
        nutrient, good_max=NUTRIENT_LOSS_BANDS["good_max"], warn_max=NUTRIENT_LOSS_BANDS["warn_max"]
    )

    return {
        "ready": True,
        "iteration": payload.get("iteration"),
        "ratio_loss": {
            "value": ratio,
            "band": ratio_band,
            "label": "Ratio loss",
            "direction": "lower_better",
            "unit": "surrogate",
            "source": ratio_src,
            "explanation": (
                "Mass-normalized deviation from neighborhood pasta∶egg ratio samples "
                f"(good ≤ {RATIO_LOSS_BANDS['good_max']}, warn ≤ {RATIO_LOSS_BANDS['warn_max']}). "
                "Blank if the chosen recipe has no ratio term."
            ),
            "thresholds": dict(RATIO_LOSS_BANDS),
        },
        "nutrient_loss": {
            "value": nutrient,
            "band": nutrient_band,
            "label": "Nutrient loss",
            "direction": "lower_better",
            "unit": "pfc_slack",
            "source": nutrient_src,
            "explanation": (
                "L1 calorie-fraction distance outside the protein/carb/fat box "
                "(0 = inside the box). Recomputed from the chosen recipe PFC."
            ),
            "thresholds": dict(NUTRIENT_LOSS_BANDS),
        },
        "holistic_0_10": {
            "value": holistic,
            "band": band_for_holistic_0_10(holistic),
            "label": "Holistic",
            "direction": "higher_better",
            "unit": "0_10",
            "source": holistic_source,
            "explanation": "LLM judge 0–10 when available; otherwise intent/composite × 10.",
        },
        "ingredients": ingredients,
        "score_history": list(payload.get("score_history") or []),
        "L_max_norm": _as_float((payload.get("diagnosis") or {}).get("L_max_norm"))
        or _as_float(((ctx.get("opt") or {}).get("term_losses") or {}).get("L_max_norm")),
        "status": payload.get("status"),
        "title": payload.get("title"),
    }


def live_scores_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build a live score snapshot after diagnose / propose (best current sample)."""
    cfg = state.get("config") or {}
    payload = {
        "opt": state.get("opt"),
        "problem": state.get("problem"),
        "chosen": {"ingredients": (state.get("chosen_recipe") or {}).get("ingredients"), "opt": state.get("opt")},
        "foodon_basis_report": state.get("foodon_basis_report")
        or (state.get("problem") or {}).get("foodon_basis_report"),
        "config": cfg,
        "macro_targets": {
            "protein_min": cfg.get("protein_min"),
            "protein_max": cfg.get("protein_max"),
            "carb_min": cfg.get("carb_min"),
            "carb_max": cfg.get("carb_max"),
            "fat_min": cfg.get("fat_min"),
            "fat_max": cfg.get("fat_max"),
        },
        "iteration": state.get("iteration"),
        "diagnosis": state.get("diagnosis"),
        "run_telemetry": state.get("run_telemetry"),
        "score_history": state.get("score_history"),
        "title": state.get("title"),
        "status": state.get("status"),
    }
    return build_display_scores(payload, ready=True)


def best_branch_scores_from_bundles(
    bundles: list[dict[str, Any]],
    *,
    iteration: int,
    box: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """One history point per branch from the best (lowest delta_L_star) LP bundle."""
    best: dict[str, dict[str, Any]] = {}
    for b in bundles or []:
        if not b.get("lp_evaluated") or b.get("oscillation_blocked"):
            continue
        branch = str(b.get("branch") or "in_distribution")
        d = _as_float(b.get("delta_L_star"))
        prev = best.get(branch)
        if prev is None or (d is not None and (prev.get("delta_L_star") is None or d < prev["delta_L_star"])):
            nutrient = _as_float(b.get("nutrient_slack"))
            if nutrient is None and box and b.get("pfc_after"):
                nutrient = _pfc_box_slack(b.get("pfc_after"), box)
            # Bundle ratio_term may be surrogate or value; prefer next_problem term_losses when present
            ratio = None
            np_ = b.get("next_problem") if isinstance(b.get("next_problem"), dict) else None
            # next_problem usually stripped in SSE; use stored ratio_term if >0 or always store
            ratio = _as_float(b.get("ratio_surrogate"))
            if ratio is None:
                # If ratio_term looks like a raw ratio (typically >1), don't use as loss
                rt = _as_float(b.get("ratio_term"))
                if rt is not None and rt < 1.5:
                    ratio = rt
            best[branch] = {
                "iteration": iteration,
                "branch": branch,
                "delta_L_star": d,
                "ratio_loss": ratio,
                "nutrient_loss": nutrient,
                "bundle_id": b.get("bundle_id"),
            }
    return list(best.values())


ID_PATH_BRANCHES = frozenset(
    {"in_distribution", "moderate", "must_retry_feasible", "accept_polish", "current"}
)
OOD_PATH_BRANCHES = frozenset({"ood_protein", "ood_other", "hybrid"})


def branch_path_family(branch: str | None) -> str | None:
    """Map a candidate branch tag to ``in_distribution`` or ``ood`` path family."""
    b = str(branch or "in_distribution")
    if b in ID_PATH_BRANCHES:
        return "in_distribution"
    if b in OOD_PATH_BRANCHES or b.startswith("ood"):
        return "ood"
    if b == "hybrid":
        return "ood"
    if b.startswith("in_"):
        return "in_distribution"
    return None


def _candidate_rank_key(candidate: dict[str, Any], *, ood_handicap: float = 0.015) -> tuple:
    """Lower is better. Prefer LP ``delta_L_star``, then diagnosis norms."""
    d = _as_float(candidate.get("delta_L_star"))
    if d is not None:
        fam = branch_path_family(candidate.get("branch"))
        if fam == "ood":
            nutrient = _as_float(candidate.get("nutrient_slack"))
            extra = 0.0
            if nutrient is not None and nutrient <= 1e-6:
                extra = 0.5 * float(ood_handicap)
            d = d - float(ood_handicap) - extra
        return (0, d)

    lmn = _as_float(candidate.get("L_max_norm"))
    if lmn is None and isinstance(candidate.get("diagnosis_full"), dict):
        lmn = _as_float(candidate["diagnosis_full"].get("L_max_norm"))
    if lmn is not None:
        n_red = int(candidate.get("n_red") or (candidate.get("diagnosis_full") or {}).get("n_red") or 99)
        obj = _as_float(candidate.get("objective"))
        if obj is None:
            obj = _as_float((candidate.get("opt") or {}).get("objective"))
        return (1, lmn, n_red, obj if obj is not None else 99.0)

    obj = _as_float((candidate.get("opt") or {}).get("objective"))
    if obj is not None:
        return (2, obj)
    return (3, 99.0)


def _candidate_has_recipe_signal(candidate: dict[str, Any]) -> bool:
    if candidate.get("ingredients"):
        return True
    if isinstance(candidate.get("next_problem"), dict):
        cr = (candidate["next_problem"].get("chosen_recipe") or {}).get("ingredients")
        if cr:
            return True
    opt = candidate.get("opt") or {}
    if opt.get("x_opt"):
        return True
    if candidate.get("x_opt"):
        return True
    return False


def _normalize_candidate_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    if not row.get("candidate_id") and row.get("bundle_id"):
        row["candidate_id"] = f"bundle::{row['bundle_id']}"
    return row


def _candidate_to_final_payload(
    candidate: dict[str, Any],
    state: dict[str, Any],
    *,
    path_key: str,
) -> dict[str, Any]:
    """Build a mini final payload for one path-family champion."""
    cfg = state.get("config") or {}
    candidate = _normalize_candidate_row(candidate)
    next_problem = candidate.get("next_problem") if isinstance(candidate.get("next_problem"), dict) else None
    problem = next_problem or state.get("problem") or {}

    opt = dict(candidate.get("opt") or {})
    if not opt.get("term_losses"):
        tl: dict[str, float] = {}
        rs = _as_float(candidate.get("ratio_surrogate"))
        if rs is not None:
            tl["ratio_surrogate"] = rs
        else:
            rt = _as_float(candidate.get("ratio_term"))
            if rt is not None and rt < 1.5:
                tl["ratio_surrogate"] = rt
        if tl:
            opt["term_losses"] = tl
    if not opt.get("pfc_after") and candidate.get("pfc_after"):
        opt["pfc_after"] = candidate["pfc_after"]
    if not opt.get("x_opt") and candidate.get("x_opt"):
        opt["x_opt"] = candidate["x_opt"]

    ingredients = candidate.get("ingredients")
    if not ingredients and next_problem:
        ingredients = (next_problem.get("chosen_recipe") or {}).get("ingredients")
    if not ingredients:
        ingredients = (state.get("chosen_recipe") or {}).get("ingredients")

    foodon = (
        candidate.get("foodon_basis_report")
        or problem.get("foodon_basis_report")
        or state.get("foodon_basis_report")
        or ((problem.get("chosen_recipe") or {}).get("foodon_basis_report"))
    )

    chosen = {
        "source": f"path_{path_key}",
        "entry": candidate,
        "opt": opt,
        "ingredients": ingredients,
        "edits": candidate.get("edits"),
        "branch": candidate.get("branch") or path_key,
        "candidate_id": candidate.get("candidate_id"),
        "delta_L_star": candidate.get("delta_L_star"),
        "foodon_basis_report": foodon,
    }
    payload: dict[str, Any] = {
        "path_key": path_key,
        "path_label": "In-distribution" if path_key == "in_distribution" else "OOD",
        "chosen": chosen,
        "problem": problem,
        "opt": opt,
        "config": cfg,
        "macro_targets": {
            "protein_min": cfg.get("protein_min"),
            "protein_max": cfg.get("protein_max"),
            "carb_min": cfg.get("carb_min"),
            "carb_max": cfg.get("carb_max"),
            "fat_min": cfg.get("fat_min"),
            "fat_max": cfg.get("fat_max"),
        },
        "foodon_basis_report": foodon,
        "title": state.get("title"),
        "iteration": candidate.get("iteration") if candidate.get("iteration") is not None else state.get("iteration"),
        "score_history": state.get("score_history") or [],
    }
    payload["display_scores"] = build_display_scores(payload)
    return payload


def select_path_finalists(
    state: dict[str, Any],
    *,
    ood_handicap: float = 0.015,
) -> dict[str, Any | None]:
    """Pick the best in-distribution and OOD path champions for dual final display."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(raw: dict[str, Any] | None) -> None:
        if not isinstance(raw, dict):
            return
        row = _normalize_candidate_row(raw)
        cid = str(row.get("candidate_id") or row.get("bundle_id") or "")
        if not cid:
            cid = f"anon::{len(seen)}"
        if cid in seen:
            return
        if not _candidate_has_recipe_signal(row):
            return
        seen.add(cid)
        candidates.append(row)

    for row in state.get("interesting_candidates") or []:
        _add(row)
    for row in state.get("candidate_pool") or []:
        _add(row)
    for b in state.get("bundles") or []:
        _add(b)
    for s in state.get("scored_finalists") or []:
        entry = s.get("entry") if isinstance(s.get("entry"), dict) else None
        nested = None
        if entry and isinstance(entry.get("metrics"), dict):
            nested = (entry.get("metrics") or {}).get("raw", {}).get("entry")
        pool_entry = nested if isinstance(nested, dict) else entry
        if isinstance(pool_entry, dict):
            merged = {**pool_entry, "candidate_id": s.get("candidate_id") or pool_entry.get("candidate_id")}
            if not merged.get("branch"):
                merged["branch"] = s.get("branch")
            _add(merged)

    last = state.get("last_applied_candidate") or {}
    current_branch = last.get("branch") or "in_distribution"
    _add(
        {
            "candidate_id": "current::final",
            "branch": current_branch,
            "opt": state.get("opt"),
            "diagnosis_full": state.get("diagnosis"),
            "L_max_norm": (state.get("diagnosis") or {}).get("L_max_norm"),
            "n_red": (state.get("diagnosis") or {}).get("n_red"),
            "objective": (state.get("opt") or {}).get("objective"),
            "ingredients": (state.get("chosen_recipe") or {}).get("ingredients"),
            "foodon_basis_report": state.get("foodon_basis_report")
            or (state.get("problem") or {}).get("foodon_basis_report"),
            "source": "current_state",
            "iteration": state.get("iteration"),
        }
    )

    by_family: dict[str, list[dict[str, Any]]] = {"in_distribution": [], "ood": []}
    for row in candidates:
        fam = branch_path_family(row.get("branch"))
        if fam:
            by_family[fam].append(row)

    out: dict[str, Any | None] = {"in_distribution": None, "ood": None}
    for fam in ("in_distribution", "ood"):
        pool = by_family[fam]
        if not pool:
            continue
        best = min(pool, key=lambda c: _candidate_rank_key(c, ood_handicap=ood_handicap))
        out[fam] = _candidate_to_final_payload(best, state, path_key=fam)
    return out
