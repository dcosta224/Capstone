"""Registry of dietary tagging dimensions (nutrient IDs, DVs, story mapping)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["lower_better", "higher_better"]

# USDA FoodData Central nutrient IDs (April 2026 export).
NUTRIENT_PROTEIN = 1003
NUTRIENT_SATURATED_FAT = 1258
NUTRIENT_FIBER = 1079
NUTRIENT_SODIUM = 1093
NUTRIENT_CALCIUM = 1087
NUTRIENT_ADDED_SUGARS = 1235
NUTRIENT_TOTAL_SUGARS = 2000

TAG_NUTRIENT_IDS = (
    NUTRIENT_PROTEIN,
    NUTRIENT_SATURATED_FAT,
    NUTRIENT_FIBER,
    NUTRIENT_SODIUM,
    NUTRIENT_CALCIUM,
    NUTRIENT_ADDED_SUGARS,
    NUTRIENT_TOTAL_SUGARS,
)

# FDA Daily Values per serving (2020 rules); used for absolute low/high labels.
USDA_DAILY_VALUES: dict[int, float] = {
    NUTRIENT_PROTEIN: 50.0,  # g
    NUTRIENT_SATURATED_FAT: 20.0,  # g
    NUTRIENT_FIBER: 28.0,  # g
    NUTRIENT_SODIUM: 2300.0,  # mg
    NUTRIENT_CALCIUM: 1300.0,  # mg
    NUTRIENT_ADDED_SUGARS: 50.0,  # g
    NUTRIENT_TOTAL_SUGARS: 50.0,  # g (fallback DV)
}

# Absolute per-serving thresholds as fraction of DV ("low" / "high" labels).
DEFAULT_LOW_DV_FRAC = 0.05
DEFAULT_HIGH_DV_FRAC = 0.20

# Corpus-relative percentile cutoffs for relative labels.
CORPUS_LOW_PERCENTILE = 25.0
CORPUS_HIGH_PERCENTILE = 75.0


@dataclass(frozen=True)
class NutrientDimension:
    slug: str
    nutrient_id: int
    unit: str
    direction: Direction
    stories: tuple[str, ...]
    dv_per_serving: float | None = None
    low_dv_frac: float = DEFAULT_LOW_DV_FRAC
    high_dv_frac: float = DEFAULT_HIGH_DV_FRAC
    fallback_nutrient_id: int | None = None

    @property
    def low_absolute_threshold(self) -> float | None:
        if self.dv_per_serving is None:
            return None
        return self.dv_per_serving * self.low_dv_frac

    @property
    def high_absolute_threshold(self) -> float | None:
        if self.dv_per_serving is None:
            return None
        return self.dv_per_serving * self.high_dv_frac


NUTRIENT_DIMENSIONS: tuple[NutrientDimension, ...] = (
    NutrientDimension(
        slug="protein",
        nutrient_id=NUTRIENT_PROTEIN,
        unit="g",
        direction="higher_better",
        stories=("osteoporosis",),
        dv_per_serving=USDA_DAILY_VALUES[NUTRIENT_PROTEIN],
        low_dv_frac=0.10,
        high_dv_frac=0.25,
    ),
    NutrientDimension(
        slug="saturated_fat",
        nutrient_id=NUTRIENT_SATURATED_FAT,
        unit="g",
        direction="lower_better",
        stories=("diabetes",),
        dv_per_serving=USDA_DAILY_VALUES[NUTRIENT_SATURATED_FAT],
    ),
    NutrientDimension(
        slug="fiber",
        nutrient_id=NUTRIENT_FIBER,
        unit="g",
        direction="higher_better",
        stories=("diabetes",),
        dv_per_serving=USDA_DAILY_VALUES[NUTRIENT_FIBER],
        low_dv_frac=0.05,
        high_dv_frac=0.15,
    ),
    NutrientDimension(
        slug="sodium",
        nutrient_id=NUTRIENT_SODIUM,
        unit="mg",
        direction="lower_better",
        stories=("diabetes",),
        dv_per_serving=USDA_DAILY_VALUES[NUTRIENT_SODIUM],
        low_dv_frac=0.05,
        high_dv_frac=0.15,
    ),
    NutrientDimension(
        slug="calcium",
        nutrient_id=NUTRIENT_CALCIUM,
        unit="mg",
        direction="higher_better",
        stories=("osteoporosis",),
        dv_per_serving=USDA_DAILY_VALUES[NUTRIENT_CALCIUM],
        low_dv_frac=0.10,
        high_dv_frac=0.25,
    ),
    NutrientDimension(
        slug="added_sugars",
        nutrient_id=NUTRIENT_ADDED_SUGARS,
        unit="g",
        direction="lower_better",
        stories=("diabetes",),
        dv_per_serving=USDA_DAILY_VALUES[NUTRIENT_ADDED_SUGARS],
        fallback_nutrient_id=NUTRIENT_TOTAL_SUGARS,
    ),
)

DIMENSION_BY_SLUG = {d.slug: d for d in NUTRIENT_DIMENSIONS}
DIMENSION_BY_NUTRIENT_ID = {d.nutrient_id: d for d in NUTRIENT_DIMENSIONS}
