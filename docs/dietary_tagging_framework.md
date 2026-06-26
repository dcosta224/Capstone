# Dietary Tagging Framework

Three-layer tagging model for ingredients and recipes.

## Layers

| Layer | Scope | Example (sodium) |
|-------|-------|------------------|
| **Absolute** | Per 100g (ingredient) or per recipe serving | 480 mg/100g; 920 mg/recipe |
| **Corpus-relative** | Percentile vs resolved recipe corpus | 72nd percentile → corpus label `high` |
| **User-relative** | Fit vs `DietaryProfile` bounds | User max 600 mg → recipe 420 mg scores well |

## Schema

Applied via [`sql/14_create_tag_schema.sql`](../sql/14_create_tag_schema.sql):

- `tag.dimension` — dimension registry (slug, nutrient_id, DV thresholds, stories)
- `tag.ingredient_nutrient` — absolute per-100g values for `food_4macro` foods
- `tag.recipe_nutrient` — recipe rollups + absolute/corpus labels
- `tag.corpus_percentile` — reference percentiles (p10, p25, p50, p75, p90)
- `tag.ingredient_restriction` / `tag.recipe_restriction` — allergen and meat/dairy flags

## Dimension registry

Defined in [`scripts/tag_dimensions.py`](../scripts/tag_dimensions.py).

| Slug | Nutrient ID | Direction | Stories | FDA DV (per serving) |
|------|-------------|-----------|---------|----------------------|
| protein | 1003 | higher_better | osteoporosis | 50 g |
| saturated_fat | 1258 | lower_better | diabetes | 20 g |
| fiber | 1079 | higher_better | diabetes | 28 g |
| sodium | 1093 | lower_better | diabetes | 2300 mg |
| calcium | 1087 | higher_better | osteoporosis | 1300 mg |
| added_sugars | 1235 (fallback 2000) | lower_better | diabetes | 50 g |

### Absolute labels

Compare per-serving value to `DV × low_dv_frac` and `DV × high_dv_frac` (defaults 5% and 20%; dimension-specific overrides in `tag_dimensions.py`).

### Corpus labels

Percentile ≤ 25 → `low` (for lower_better nutrients) or `low` when ≤ p25 for higher_better means below corpus norm. Percentile ≥ 75 → `high` for lower_better (worse) or higher_better (better).

## Loaders

```bash
# Apply schema once
psql $DATABASE_URL -f sql/14_create_tag_schema.sql

# Nutrient tags (ingredient + recipe)
uv run python scripts/tag_nutrients.py --execute

# Restriction / allergen tags
uv run python scripts/tag_restrictions.py --execute
```

## Restriction taxonomy

[`data/allergen_taxonomy.json`](../data/allergen_taxonomy.json) — FDA Big 9 allergens plus meat/poultry for meat-free tagging. Keyword matching on `food.description` and `branded_food.ingredients`. FoodOn ancestor IDs are reserved for future ontology integration.

Recipe rollup rule: **recipe contains restriction R if any resolved ingredient line triggers R**.

## User-relative scoring (MVP)

[`scripts/mvp_dietary_fit.py`](../scripts/mvp_dietary_fit.py) defines `DietaryProfile` with optional nutrient bounds and `required_free_labels` (e.g. `gluten_free`, `dairy_free`).

Pass to `UserQuery.dietary_profile` in [`scripts/mvp_pipeline.py`](../scripts/mvp_pipeline.py). Stage-1 ranker blends semantic + PFC + dietary scores via `w_semantic`, `w_nutrient`, and `DietaryProfile.w_dietary`.

Hard filters: recipes violating `required_free_labels` get dietary score 0 and rank lower.

## Data flow

```
food_nutrient ──► tag.ingredient_nutrient (per fdc_id)
resolved_recipes + gram_weight ──► tag.recipe_nutrient (rollup + percentiles)
food_4macro descriptions ──► tag.ingredient_restriction (keywords)
resolved_recipes ──► tag.recipe_restriction (rollup)
tag.recipe_nutrient + DietaryProfile ──► mvp_recipe_ranker dietary_score
```
