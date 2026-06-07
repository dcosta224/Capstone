# Recipe matching workspace (10K sample)

Artifacts for `scratch/food_mvp_recipe_matching.ipynb`.

## One-time (parse + embed)

Each recipe ingredient line and each `food_4macro` description gets **three** MiniLM embeddings:

| File | Description |
|------|-------------|
| `recipe_ingredients_parsed.parquet` | Parsed recipe lines + `dequantified` column |
| `recipe_name_embeddings.npy` | Parsed **name** |
| `recipe_prep_embeddings.npy` | Parsed **preparation** |
| `recipe_dequant_embeddings.npy` | Name/size/prep without quantity |
| `food_4macro_parsed.parquet` | Parsed USDA descriptions + `dequantified` |
| `food_4macro_name_embeddings.npy` | Food **name** |
| `food_4macro_prep_embeddings.npy` | Food **preparation** |
| `food_4macro_dequant_embeddings.npy` | Food dequantified text |
| `embeddings_meta.json` | Model name, row counts, dims |
| `unprepared_prep_embedding.npy` | Single vector; recipe lines with empty `preparation` use this for prep match |

Recipe ingredients without an explicit preparation default to the **unprepared** embedding (implicit raw ingredient). USDA `food_4macro` rows still embed parsed preparation from the catalog description.

Delete these to force a rebuild, or run:

```bash
uv run python scripts/ingredient_query_cache.py --rerun   # recipe + food_4macro
uv run python scripts/ingredient_query_cache.py --food-only  # DB catalog only (if recipe cache exists)
```

Default run embeds **both** ~75K recipe lines and all `food_4macro` rows. A recipe-only failure leaves food unbuilt — use `--food-only` to finish without re-embedding recipes.

Changing `RECIPE_NROWS` also requires rebuild (`--recipe-nrows`).

Legacy `food_4macro_desc_embeddings.npy` (single vector) is no longer used.

## Matching outputs

| File | Description |
|------|-------------|
| `ingredient_matches_staged.csv` | Staged hybrid match results |
| `recipe_match_summary_staged.csv` | Per-recipe tier percentages |

Delete match CSVs to re-run §3–5 after tuning `StagedMatchConfig`.

## HP sweep (staged grid search)

Two-phase grid with **per-stage** evaluation (not a single blended score):

1. **Identity** — `name_sem` × `dequant_sem`; rank by `stage1_avg` (`base_score`)
2. **Prep** — `prep_sem` with best identity fixed; rank by `stage2_avg` (`prep_score`)

| Output | Description |
|--------|-------------|
| `hp_sweep/hp_identity_leaderboard.csv` | Phase 1 results |
| `hp_sweep/hp_prep_leaderboard.csv` | Phase 2 results |
| `hp_sweep/hp_best_config.json` | Winning weights + path to best match CSV |
| `hp_sweep/ingredient_matches_identity_<slug>.csv` | Per identity-config matches |
| `hp_sweep/ingredient_matches_prep_<slug>.csv` | Per prep-config matches |

```bash
uv run python scripts/ingredient_match_hp_sweep.py --quick   # 9 + 3 = 12 match runs
uv run python scripts/ingredient_match_hp_sweep.py             # 25 + 5 = 30 match runs
```
