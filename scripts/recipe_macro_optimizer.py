"""Convex recipe portion optimizer via log-space SCA + CVXPY.

PFC gram bounds are derived at the midpoint kcal target. Total kcal is enforced
as an equality constraint at that midpoint (not a separate min/max range).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

from mvp_nutrient_fit import clamp_fraction_bounds, kcal_target_midpoint

MACRO_NAMES = ("protein_g", "fat_g", "carbs_g", "energy_kcal")
NUTRIENT_IDS = (1003, 1004, 1005, 1008)
KCAL_TOL = 1.0  # grams / kcal tolerance for "already feasible"
PFC_TOL = 0.5  # g tolerance on protein/fat/carbs

_SOLVER_CHAIN = (cp.OSQP, cp.SCS)


@dataclass
class MacroBounds:
    protein_g: tuple[float, float]
    fat_g: tuple[float, float]
    carbs_g: tuple[float, float]
    kcal_target: float

    def pfc_min_max_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        mins = np.array([self.protein_g[0], self.fat_g[0], self.carbs_g[0]])
        maxs = np.array([self.protein_g[1], self.fat_g[1], self.carbs_g[1]])
        return mins, maxs


@dataclass
class IngredientConstraint:
    locked: bool = False
    min_grams: float | None = None
    max_grams: float | None = None
    weight: float = 1.0


@dataclass
class OptimizerConfig:
    macro_bounds: MacroBounds
    macro_penalty: float = 10.0
    max_sca_iters: int = 50
    trust_region: float = 0.5
    min_grams: float = 1e-3
    sca_tol: float = 1e-4


@dataclass
class OptimizationResult:
    x_opt: np.ndarray
    r: np.ndarray
    macros_before: np.ndarray
    macros_after: np.ndarray
    portion_score: float
    avg_pct_change: float
    max_pct_change: float
    status: str
    sca_iters: int
    macro_slack: np.ndarray
    converged: bool
    macro_feasible: bool = True
    feasibility_message: str | None = None
    used_fallback: bool = False
    already_feasible: bool = False
    kcal_target: float | None = None
    constraint_violations: dict[str, bool] | None = None


@dataclass
class IngredientMeta:
    ingredient_idx: int
    ingredient: str
    fdc_id: int | None = None
    quantity: float | None = None
    unit: str | None = None
    portion_label: str | None = None
    fdc_description: str | None = None


def build_macro_matrix(
    gram_weights: np.ndarray,
    nutrient_amounts_per_100g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-gram macro matrix M (4 x n) from USDA per-100g values (n x 4)."""
    x0 = np.asarray(gram_weights, dtype=float)
    amounts = np.asarray(nutrient_amounts_per_100g, dtype=float)
    if amounts.ndim == 1:
        amounts = amounts.reshape(1, -1)
    if amounts.shape[1] != 4:
        raise ValueError(f"Expected (n, 4) per-100g nutrients; got {amounts.shape}")
    if amounts.shape[0] != len(x0):
        raise ValueError(f"Row count {amounts.shape[0]} != ingredient count {len(x0)}")
    per_gram = (amounts / 100.0).T  # (4, n)
    return x0, per_gram


def compute_macros(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ np.asarray(x, dtype=float)


def portion_adjustment_score(r: np.ndarray, weights: np.ndarray) -> float:
    """Sum w_i * (log r_i)^2. Zero when all r_i = 1 (vs chosen reference portions)."""
    r = np.asarray(r, dtype=float)
    weights = np.asarray(weights, dtype=float)
    log_r = np.log(np.maximum(r, 1e-12))
    return float(np.sum(weights * log_r**2))


def kcal_scaled_baseline(
    x0: np.ndarray,
    kcal_before: float,
    kcal_target: float,
) -> np.ndarray:
    """Uniform scale of original portions to hit target kcal (ratios unchanged)."""
    x0 = np.asarray(x0, dtype=float)
    if kcal_before <= 1e-9:
        return x0.copy()
    return x0 * (kcal_target / kcal_before)


def portion_adjustment_metrics(r: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Human-readable portion change stats from scale factors r_i (vs chosen reference).

    avg_pct_change: weighted mean of |r_i - 1| * 100 (typical % change per ingredient).
    portion_score: sum w_i * (log r_i)^2 (log-ratio penalty for the supplied reference).
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    w_norm = w / max(w.sum(), 1e-12)

    pct_changes = np.abs(r - 1.0) * 100.0
    log_r = np.log(np.maximum(r, 1e-12))

    return {
        "portion_score": float(np.sum(w * log_r**2)),
        "avg_pct_change": float(np.sum(w_norm * pct_changes)),
        "max_pct_change": float(np.max(pct_changes)),
        "min_scale_factor": float(np.min(r)),
        "max_scale_factor": float(np.max(r)),
        "rms_log_deviation": float(np.sqrt(np.sum(w_norm * log_r**2))),
    }


def baseline_scale_factors(
    x_opt: np.ndarray,
    x0: np.ndarray,
    kcal_before: float,
    kcal_target: float,
    *,
    min_grams: float = 1e-3,
) -> np.ndarray:
    """r_i = x_opt_i / x_ref_i where x_ref uniformly scales x0 to kcal_target."""
    x_ref = kcal_scaled_baseline(x0, kcal_before, kcal_target)
    return np.asarray(x_opt, dtype=float) / np.maximum(x_ref, min_grams)


def display_portion_change_metrics(
    x_opt: np.ndarray,
    x0: np.ndarray,
    kcal_before: float,
    kcal_target: float,
    weights: np.ndarray | None = None,
    *,
    min_grams: float = 1e-3,
) -> dict[str, float]:
    """Portion change vs uniform kcal-scaled baseline (0 when only calories are scaled)."""
    r = baseline_scale_factors(x_opt, x0, kcal_before, kcal_target, min_grams=min_grams)
    metrics = portion_adjustment_metrics(r, weights)
    return {
        "portion_score": metrics["portion_score"],
        "avg_pct_change": metrics["avg_pct_change"],
        "max_pct_change": metrics["max_pct_change"],
        "min_scale_factor": metrics["min_scale_factor"],
        "max_scale_factor": metrics["max_scale_factor"],
    }


def derive_macro_bounds_from_fractions(
    kcal_min: float,
    kcal_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    protein_frac_min: float,
    protein_frac_max: float,
) -> MacroBounds:
    """PFC gram bounds at midpoint kcal; kcal enforced separately as equality."""
    f0, f1, c0, c1, p0, p1 = clamp_fraction_bounds(
        fat_frac_min,
        fat_frac_max,
        carb_frac_min,
        carb_frac_max,
        protein_frac_min,
        protein_frac_max,
    )
    kcal_mid = kcal_target_midpoint(kcal_min, kcal_max)
    return MacroBounds(
        protein_g=(kcal_mid * p0 / 4, kcal_mid * p1 / 4),
        fat_g=(kcal_mid * f0 / 9, kcal_mid * f1 / 9),
        carbs_g=(kcal_mid * c0 / 4, kcal_mid * c1 / 4),
        kcal_target=kcal_mid,
    )


def constraint_violations(
    macros: np.ndarray,
    bounds: MacroBounds,
) -> dict[str, bool]:
    t_min, t_max = bounds.pfc_min_max_arrays()
    names = ("protein_g", "fat_g", "carbs_g", "energy_kcal")
    out: dict[str, bool] = {}
    for j, name in enumerate(names[:3]):
        out[name] = bool(macros[j] < t_min[j] - PFC_TOL or macros[j] > t_max[j] + PFC_TOL)
    out["energy_kcal"] = bool(abs(macros[3] - bounds.kcal_target) > KCAL_TOL)
    return out


def is_already_feasible(macros: np.ndarray, bounds: MacroBounds) -> bool:
    violations = constraint_violations(macros, bounds)
    return not any(violations.values())


def _solve_problem(prob: cp.Problem) -> str:
    for solver in _SOLVER_CHAIN:
        try:
            kwargs: dict[str, Any] = {"verbose": False}
            if solver == cp.OSQP:
                kwargs["warm_start"] = True
            prob.solve(solver=solver, **kwargs)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.status
        except cp.SolverError:
            continue
    return prob.status or "solver_failed"


def check_macro_region_feasible(
    M: np.ndarray,
    macro_bounds: MacroBounds,
) -> tuple[bool, np.ndarray | None, str]:
    """LP: exists x >= 0 with PFC in box and total kcal = kcal_target."""
    M = np.asarray(M, dtype=float)
    n = M.shape[1]
    t_min, t_max = macro_bounds.pfc_min_max_arrays()

    if n == 0:
        return False, None, "Recipe has no ingredients."

    x = cp.Variable(n, nonneg=True)
    cons: list[Any] = []
    for j in range(3):
        expr = M[j] @ x
        cons.append(expr >= t_min[j])
        cons.append(expr <= t_max[j])
    cons.append(M[3] @ x == macro_bounds.kcal_target)

    prob = cp.Problem(cp.Minimize(0), cons)
    status = _solve_problem(prob)

    if status in ("optimal", "optimal_inaccurate") and x.value is not None:
        witness = np.asarray(x.value, dtype=float).ravel()
        return (
            True,
            witness,
            "A non-negative portion mix can reach the PFC targets at the goal kcal.",
        )

    return (
        False,
        None,
        "No non-negative combination of these ingredients can hit the PFC targets "
        f"at {macro_bounds.kcal_target:.0f} kcal.",
    )


def make_fallback_result(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    message: str,
    status: str = "infeasible_region",
    bounds: MacroBounds | None = None,
) -> OptimizationResult:
    macros = compute_macros(x0, M)
    n = len(x0)
    weights = np.ones(n, dtype=float)
    kcal_tgt = bounds.kcal_target if bounds else float(macros[3])
    display = display_portion_change_metrics(x0, x0, float(macros[3]), kcal_tgt, weights)
    return OptimizationResult(
        x_opt=x0.copy(),
        r=np.ones(n),
        macros_before=macros,
        macros_after=macros.copy(),
        portion_score=display["portion_score"],
        avg_pct_change=display["avg_pct_change"],
        max_pct_change=display["max_pct_change"],
        status=status,
        sca_iters=0,
        macro_slack=np.zeros(4),
        converged=False,
        macro_feasible=False,
        feasibility_message=message,
        used_fallback=True,
        already_feasible=False,
        kcal_target=bounds.kcal_target if bounds else None,
        constraint_violations=constraint_violations(macros, bounds) if bounds else None,
    )


def make_already_feasible_result(
    x0: np.ndarray,
    M: np.ndarray,
    bounds: MacroBounds,
) -> OptimizationResult:
    macros = compute_macros(x0, M)
    n = len(x0)
    weights = np.ones(n, dtype=float)
    display = display_portion_change_metrics(
        x0, x0, float(macros[3]), bounds.kcal_target, weights
    )
    return OptimizationResult(
        x_opt=x0.copy(),
        r=np.ones(n),
        macros_before=macros,
        macros_after=macros.copy(),
        portion_score=display["portion_score"],
        avg_pct_change=display["avg_pct_change"],
        max_pct_change=display["max_pct_change"],
        status="already_feasible",
        sca_iters=0,
        macro_slack=np.zeros(4),
        converged=True,
        macro_feasible=True,
        feasibility_message=(
            f"Recipe already meets PFC targets at {bounds.kcal_target:.0f} kcal; "
            "portions unchanged."
        ),
        used_fallback=False,
        already_feasible=True,
        kcal_target=bounds.kcal_target,
        constraint_violations=constraint_violations(macros, bounds),
    )


class RecipeMacroOptimizer:
    def optimize(
        self,
        x0: np.ndarray,
        M: np.ndarray,
        config: OptimizerConfig,
        constraints: list[IngredientConstraint] | None = None,
    ) -> OptimizationResult:
        x0 = np.asarray(x0, dtype=float)
        M = np.asarray(M, dtype=float)
        n = len(x0)
        if M.shape != (4, n):
            raise ValueError(f"M must be shape (4, n); got {M.shape} for n={n}")

        bounds = config.macro_bounds
        constraints = constraints or [IngredientConstraint() for _ in range(n)]
        if len(constraints) != n:
            raise ValueError("constraints length must match number of ingredients")

        macros_before = compute_macros(x0, M)
        if is_already_feasible(macros_before, bounds):
            return make_already_feasible_result(x0, M, bounds)

        feasible, _, feas_msg = check_macro_region_feasible(M, bounds)
        if not feasible:
            return make_fallback_result(x0, M, message=feas_msg, bounds=bounds)

        weights = np.array([c.weight for c in constraints], dtype=float)
        u0 = np.log(np.maximum(x0, config.min_grams))
        x_ref = kcal_scaled_baseline(x0, float(macros_before[3]), bounds.kcal_target)
        u_ref = np.log(np.maximum(x_ref, config.min_grams))
        t_min, t_max = bounds.pfc_min_max_arrays()
        kcal_target = bounds.kcal_target

        u_curr = u0.copy()
        macro_slack = np.zeros(4)
        status = "incomplete"
        converged = False

        for it in range(config.max_sca_iters):
            x_curr = np.exp(u_curr)
            g_curr = compute_macros(x_curr, M)
            grad = M * x_curr[np.newaxis, :]

            u = cp.Variable(n)
            s_plus = cp.Variable(3, nonneg=True)
            s_minus = cp.Variable(3, nonneg=True)
            s_kcal = cp.Variable(nonneg=True)

            objective = cp.Minimize(
                cp.sum(cp.multiply(weights, cp.square(u - u_ref)))
                + config.macro_penalty * (cp.sum(s_plus + s_minus) + s_kcal)
            )

            cons = []
            for j in range(3):
                lin = g_curr[j] + grad[j] @ (u - u_curr)
                cons.append(lin >= t_min[j] - s_minus[j])
                cons.append(lin <= t_max[j] + s_plus[j])

            lin_kcal = g_curr[3] + grad[3] @ (u - u_curr)
            cons.append(lin_kcal >= kcal_target - s_kcal)
            cons.append(lin_kcal <= kcal_target + s_kcal)

            for i, c in enumerate(constraints):
                if c.locked:
                    cons.append(u[i] == u0[i])
                else:
                    lo = np.log(config.min_grams)
                    hi = np.log(np.maximum(x0[i] * 100, config.min_grams * 100))
                    if c.min_grams is not None:
                        lo = max(lo, np.log(c.min_grams))
                    if c.max_grams is not None:
                        hi = min(hi, np.log(c.max_grams))
                    cons.append(u[i] >= lo)
                    cons.append(u[i] <= hi)

            cons.append(cp.abs(u - u_curr) <= config.trust_region)

            prob = cp.Problem(objective, cons)
            status = _solve_problem(prob)

            if u.value is None:
                return make_fallback_result(
                    x0,
                    M,
                    message=(
                        "Optimizer could not improve portions for this recipe; "
                        "original amounts kept."
                    ),
                    status="solver_failed",
                    bounds=bounds,
                )

            u_new = np.asarray(u.value, dtype=float).ravel()
            delta = float(np.max(np.abs(u_new - u_curr)))
            u_curr = u_new
            pfc_slack = (
                float(cp.sum(s_plus + s_minus).value)
                if s_plus.value is not None
                else 0.0
            )
            kcal_slack = float(s_kcal.value) if s_kcal.value is not None else 0.0
            macro_slack = np.array([0.0, 0.0, 0.0, kcal_slack])
            if pfc_slack:
                macro_slack[:3] = pfc_slack / 3.0

            if delta < config.sca_tol:
                converged = True
                break

        x_opt = np.exp(u_curr)
        r = x_opt / np.maximum(x0, config.min_grams)
        macros_after = compute_macros(x_opt, M)
        display = display_portion_change_metrics(
            x_opt,
            x0,
            float(macros_before[3]),
            kcal_target,
            weights,
            min_grams=config.min_grams,
        )

        return OptimizationResult(
            x_opt=x_opt,
            r=r,
            macros_before=macros_before,
            macros_after=macros_after,
            portion_score=display["portion_score"],
            avg_pct_change=display["avg_pct_change"],
            max_pct_change=display["max_pct_change"],
            status=status,
            sca_iters=it + 1,
            macro_slack=macro_slack,
            converged=converged,
            macro_feasible=True,
            feasibility_message=feas_msg,
            used_fallback=False,
            already_feasible=False,
            kcal_target=kcal_target,
            constraint_violations=constraint_violations(macros_after, bounds),
        )


def format_serving_display(
    result: OptimizationResult,
    x0: np.ndarray,
    meta: list[IngredientMeta],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(meta):
        qty_orig = m.quantity
        qty_opt = None
        if qty_orig is not None and qty_orig > 0:
            qty_opt = float(qty_orig * result.r[i])
        rows.append(
            {
                "ingredient_idx": m.ingredient_idx,
                "ingredient": m.ingredient,
                "fdc_id": m.fdc_id,
                "fdc_description": m.fdc_description,
                "portion_label": m.portion_label,
                "unit": m.unit,
                "quantity_original": qty_orig,
                "quantity_optimized": qty_opt,
                "gram_weight_original": float(x0[i]),
                "gram_weight_optimized": float(result.x_opt[i]),
                "adjustment_factor": float(result.r[i]),
            }
        )
    return rows


def macros_to_dict(macros: np.ndarray) -> dict[str, float]:
    return {name: float(macros[j]) for j, name in enumerate(MACRO_NAMES)}
