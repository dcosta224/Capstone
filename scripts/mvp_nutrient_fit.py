"""PFC calorie-fraction nutrient range fit score."""

from __future__ import annotations

import numpy as np

PFC_NAMES = ("fat", "carbs", "protein")
FRAC_EPS = 1e-6


def clamp_fraction(value: float) -> float:
    """Map 0 (or negative) lower bounds to a small positive epsilon."""
    return max(float(value), FRAC_EPS)


def clamp_fraction_bounds(
    fat_frac_min: float,
    fat_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    protein_frac_min: float,
    protein_frac_max: float,
) -> tuple[float, float, float, float, float, float]:
    return (
        clamp_fraction(fat_frac_min),
        float(fat_frac_max),
        clamp_fraction(carb_frac_min),
        float(carb_frac_max),
        clamp_fraction(protein_frac_min),
        float(protein_frac_max),
    )


def kcal_target_midpoint(kcal_min: float, kcal_max: float) -> float:
    return (float(kcal_min) + float(kcal_max)) / 2.0


def pfc_calorie_fractions(
    fat_g: float,
    carbs_g: float,
    protein_g: float,
    energy_kcal: float,
) -> np.ndarray:
    """Return [fat_frac, carb_frac, protein_frac] of calories from each macro."""
    if energy_kcal <= 0:
        raise ValueError("energy_kcal must be positive")
    return np.array(
        [
            (fat_g * 9.0) / energy_kcal,
            (carbs_g * 4.0) / energy_kcal,
            (protein_g * 4.0) / energy_kcal,
        ],
        dtype=float,
    )


def nutrient_fit_to_score(fit: float, *, in_range: bool = False, tol: float = 1e-6) -> float:
    """Map PFC fit distance to [0, 1] score (1 = in range, 0 = arbitrarily poor)."""
    if in_range or fit <= tol:
        return 1.0
    if not np.isfinite(fit) or fit <= 0:
        return 0.0
    return float(1.0 / (1.0 + fit))


def nutrient_range_fit(p: np.ndarray, L: np.ndarray, U: np.ndarray) -> float:
    """Log-ratio distance from feasible box. Zero when p in [L, U] element-wise."""
    p = np.asarray(p, dtype=float)
    L = np.maximum(np.asarray(L, dtype=float), FRAC_EPS)
    U = np.asarray(U, dtype=float)
    p_safe = np.maximum(p, FRAC_EPS)
    if np.any(U <= 0):
        return float("inf")
    violations = np.maximum(0.0, np.maximum(np.log(L / p_safe), np.log(p_safe / U)))
    return float(np.linalg.norm(violations))


def nutrient_range_fit_from_totals(
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    energy_kcal: float,
    fat_frac_min: float,
    fat_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    protein_frac_min: float,
    protein_frac_max: float,
) -> float:
    f0, f1, c0, c1, p0, p1 = clamp_fraction_bounds(
        fat_frac_min,
        fat_frac_max,
        carb_frac_min,
        carb_frac_max,
        protein_frac_min,
        protein_frac_max,
    )
    p = pfc_calorie_fractions(fat_g, carbs_g, protein_g, energy_kcal)
    L = np.array([f0, c0, p0])
    U = np.array([f1, c1, p1])
    return nutrient_range_fit(p, L, U)


def pfc_in_range(
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    energy_kcal: float,
    fat_frac_min: float,
    fat_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    protein_frac_min: float,
    protein_frac_max: float,
    *,
    tol: float = 1e-6,
) -> bool:
    return (
        nutrient_range_fit_from_totals(
            protein_g,
            fat_g,
            carbs_g,
            energy_kcal,
            fat_frac_min,
            fat_frac_max,
            carb_frac_min,
            carb_frac_max,
            protein_frac_min,
            protein_frac_max,
        )
        <= tol
    )
