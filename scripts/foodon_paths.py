"""Local paths for FoodOn ontology files (no download required)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOODON_REPO = ROOT / "Data" / "foodon-master"
FOODON_OWL = FOODON_REPO / "foodon.owl"
FOODON_SYNONYMS_TSV = FOODON_REPO / "foodon-synonyms.tsv"
FOODON_INDEX_CACHE = ROOT / "scratch" / "foodon_index.json"
FOODON_WEB_CACHE = ROOT / "foodon_web" / "cache" / "foodon_index.json"

USDA_DATA_DIR = ROOT / "Data" / "All_Food_Data_April_2026"
USDA_FOOD_CSV = USDA_DATA_DIR / "food.csv"
USDA_BRANDED_CSV = USDA_DATA_DIR / "branded_food.csv"


def resolve_owl_path() -> Path:
    if FOODON_OWL.is_file():
        return FOODON_OWL
    fallback = FOODON_REPO / "foodon-base.owl"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"FoodOn OWL not found. Expected {FOODON_OWL} (clone or unzip foodon into Data/foodon-master)."
    )
