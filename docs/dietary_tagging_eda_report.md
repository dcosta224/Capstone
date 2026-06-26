# Dietary Tagging EDA — GitHub Issue Comment

> Copy/paste this comment onto the precise food tagging issue after running
> `scratch/EDA/dietary_tagging_eda.ipynb` (or `uv run python scripts/tag_eda.py`)
> against a loaded Supabase database.

## Summary

EDA on `usda.food_4macro` and `recipe.resolved_recipes` for diabetes, osteoporosis, and dietary-restriction tagging dimensions.

### Nutrient coverage on `food_4macro` (~97k foods)

| Nutrient ID | Name | Expected coverage | Tagging recommendation |
|-------------|------|-------------------|------------------------|
| 1003 | Protein | ~100% (required for food_4macro) | absolute + corpus_relative + user_relative |
| 1258 | Saturated fat | High on foundation + branded | absolute + corpus_relative |
| 1079 | Fiber | Moderate–high | absolute + corpus_relative |
| 1093 | Sodium | High | absolute + corpus_relative |
| 1087 | Calcium | Moderate | absolute + corpus_relative |
| 1235 | Added sugars | **Sparse** on foundation foods | use fallback total sugars (2000) where added missing |
| 2000 | Total sugars | Moderate (fallback) | absolute only when 1235 absent |

**Action:** Run notebook Section 1 SQL to fill actual `%` columns from your DB load.

### Resolved recipe corpus

- Tagging is limited to recipes in `recipe.resolved_recipes` with non-null `fdc_id` and `gram_weight`.
- MVP corpus is ~106 fully resolved recipes today; corpus percentiles should be recomputed as resolution expands.
- Per-dimension recipe coverage = % of resolved lines with nutrient value × rollup success.

### Restriction / allergen feasibility

| Source | Coverage | Notes |
|--------|----------|-------|
| `fdc_description` keyword match | All food_4macro | Deterministic; false positives possible on compound words |
| `branded_food.ingredients` | Branded subset only | Better for packaged foods |
| FoodOn ontology | **Not yet on branch** | `foodon_index.py` missing; keyword taxonomy ships first |

Implemented taxonomy: `data/allergen_taxonomy.json` (FDA Big 9 + meat/poultry).

### Gaps and risks

1. **Added sugars** — expect low coverage on SR Legacy / Foundation; fallback to total sugars documented in `tag_dimensions.py`.
2. **Gluten-free** — keyword `wheat`/`flour`/`pasta` heuristic; does not catch barley/rye without FoodOn or expanded keywords.
3. **Vegetarian** — mapped to `meat_free` via meat + poultry keyword rules; not equivalent to strict vegetarian (e.g. fish still allowed unless `fish_free` required).
4. **Cuisine tagging** — deprioritized per issue; no EDA in this pass.

### Implementation status (this branch)

- [x] `sql/14_create_tag_schema.sql` — tag schema
- [x] `scripts/tag_nutrients.py` — ingredient + recipe nutrient tags
- [x] `scripts/tag_restrictions.py` — allergen/restriction tags
- [x] `scripts/mvp_dietary_fit.py` — user-relative scoring
- [x] `docs/dietary_tagging_framework.md` — framework design

### Next steps

1. Run EDA notebook against production DB and replace placeholder coverage table above.
2. Sync FoodOn files from foodon branch for ontology-backed restrictions.
3. Expand keyword lists from EDA false-positive/false-negative review.
4. Wire `tag.recipe_nutrient` columns into MVP corpus cache for live dietary ranking.
