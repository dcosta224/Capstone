# Portion resolution roadmap

Multi-signal resolution plans replace single `amount_kind` classification. This document tracks what is implemented vs deferred.

## Implemented

- [x] **`resolution_plan.py`** — `ResolutionPlan` with ordered `resolution_paths`, terminal `flags`, parenthetical mass extraction, size/count tokens
- [x] **Gram priority ladder** — `resolve_grams_from_plan()` in `portion_gram.py` with statuses: `ok_embedded_mass`, `unresolvable_serving_only`, `ambiguous_accepted`, `vague_amount`, `compound_skipped`
- [x] **Count size/modifier matching** — `CountPortionCandidate.score_for` uses size tokens in modifier text (e.g. `small (3" long)`)
- [x] **Serving-only detector** — terminal `unresolvable_serving_only` when only mass/serving rows exist
- [x] **`portion_candidate_index.py`** — per-fdc portion summaries + `portion_match_score`
- [x] **Portion-informed retrieval** — mass-in-text → semantic top-10; else semantic pool ranked by `0.6 * retrieval + 0.4 * portion_match`
- [x] **Judge schema** — optional `matched_portion_id` wired into gram resolve
- [x] **`line_enrichment_llm.py`** — selective LLM enrichment for compound, vague, ambiguous, parenthetical cases
- [x] **Golden fixtures** — `tests/fixtures/portion_resolution_cases.json`

## Deferred (not now)

- LLM gram-weight guess from parametric knowledge for `ambiguous_quantity_accepted` (e.g. "small box pretzels")
- Full compound-ingredient split with per-component fdc match and calorie rollup
- Default priors for vague amounts ("bit of garlic" → 1 clove / 3g)
- User-configurable portion assumptions and manual overrides in UI
- Cross-database fallback (Open Food Facts) when USDA has no item portion
- Learned ranker replacing hand-tuned `0.6/0.4` retrieval blend

## Metrics to track

Re-run `scripts/portion_pipeline_feasibility.py` on the 1k sample (seed 42) and compare:

| Metric | Notes |
|--------|-------|
| `fdc_and_gram_rate_all` | Overall fdc + grams |
| `fdc_and_gram_rate_count` / `_volume` / `_mass` | By primary path |
| `rules_grams_status_counts` | New terminal statuses vs false `no_portion` |
| `resolution_path_counts` | From amount classification phase |
| `plan_flag_counts` | Explicit terminal flags |

```bash
uv run python scripts/portion_pipeline_feasibility.py --n-recipes 1000 --seed 42
uv run python scripts/portion_pipeline_feasibility.py --force-payloads   # new retrieval prompts
uv run python scripts/portion_pipeline_feasibility.py --force-judging    # re-judge (costly)
```
