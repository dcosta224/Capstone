#!/usr/bin/env python3
"""List unique units on 10K RecipeNLG lines matching chicken + breast (substring filter)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ingredient_parser import parse_ingredient

from ingredient_query_cache import DEFAULT_RECIPE_CSV, DEFAULT_RECIPE_NROWS, load_recipe_ingredients
from parse_recipe_ingredient import _flatten_amounts, parse_ingredient_fields

ROOT = Path(__file__).resolve().parents[1]


def matches_chicken_breast(text: str) -> bool:
    lower = text.casefold()
    return "hicken" in lower and "reast" in lower


def units_from_ingredient(text: str) -> list[str]:
    """All normalized unit strings from every parsed amount on the line."""
    try:
        parsed = parse_ingredient(
            text,
            string_units=True,
            volumetric_units_system="us_customary",
        )
    except Exception:
        return []

    units: list[str] = []
    for amount in _flatten_amounts(parsed.amount or []):
        unit = getattr(amount, "unit", None)
        if unit:
            units.append(str(unit).strip())
    return units


def primary_unit(text: str) -> str | None:
    return parse_ingredient_fields(text).get("unit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find RecipeNLG ingredient lines containing both 'hicken' and 'reast' "
            "(case-insensitive substrings), parse units, print unique values."
        ),
    )
    parser.add_argument(
        "--recipe-csv",
        type=Path,
        default=DEFAULT_RECIPE_CSV,
        help=f"RecipeNLG CSV (default: {DEFAULT_RECIPE_CSV})",
    )
    parser.add_argument(
        "--recipe-nrows",
        type=int,
        default=DEFAULT_RECIPE_NROWS,
        help="Number of recipes to load (default: 10000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full summary as JSON (counts + unit lists)",
    )
    args = parser.parse_args()

    if not args.recipe_csv.is_file():
        raise SystemExit(f"Recipe CSV not found: {args.recipe_csv}")

    ingredients = load_recipe_ingredients(recipe_csv=args.recipe_csv, nrows=args.recipe_nrows)
    hits = ingredients[ingredients["ingredient"].map(matches_chicken_breast)]
    n_hits = len(hits)
    print(f"Recipe ingredient lines (first {args.recipe_nrows:,} recipes): {len(ingredients):,}")
    print(f"Lines with 'hicken' and 'reast': {n_hits:,}\n")

    all_units: list[str] = []
    primary_units: list[str] = []
    no_unit = 0

    for text in hits["ingredient"].astype(str):
        line_units = units_from_ingredient(text)
        if line_units:
            all_units.extend(line_units)
        else:
            no_unit += 1
        pu = primary_unit(text)
        if pu:
            primary_units.append(str(pu).strip())

    unique_all = sorted(set(all_units))
    unique_primary = sorted(set(primary_units))
    counts_all = Counter(all_units)
    counts_primary = Counter(primary_units)

    if args.json:
        payload = {
            "n_ingredient_lines": len(ingredients),
            "n_chicken_breast_lines": n_hits,
            "n_lines_without_parsed_unit": no_unit,
            "unique_units_all_amounts": unique_all,
            "unit_counts_all_amounts": dict(counts_all.most_common()),
            "unique_units_primary_amount": unique_primary,
            "unit_counts_primary_amount": dict(counts_primary.most_common()),
        }
        print(json.dumps(payload, indent=2))
        return

    print("Unique units (every parsed amount on matching lines):")
    for u in unique_all:
        print(f"  {u!r}  ({counts_all[u]:,}×)")
    print(f"\nTotal unit tokens: {len(all_units):,}  |  Lines with no parsed unit: {no_unit:,}")

    print("\nUnique primary units (first amount with a unit, same as parse_ingredient_fields['unit']):")
    for u in unique_primary:
        print(f"  {u!r}  ({counts_primary[u]:,}×)")


if __name__ == "__main__":
    main()
