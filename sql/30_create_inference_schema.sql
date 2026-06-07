-- LLM ingredient-matching storage.
-- Schema `inference` is organized so that ONE EXECUTION OF THE SCRIPT = ONE
-- EXPERIMENT:
--   * inference.match_experiments_0  one row per script execution (the experiment)
--   * inference.match_inferences_0   one row per individual LLM call (inference)
--   * inference.match_candidates_0   top-N retrieval near-misses per inference
--   * inference.spend_checks_0       budget circuit-breaker audit trail
-- Idempotent: safe to re-run; uses IF NOT EXISTS and never drops data.

CREATE SCHEMA IF NOT EXISTS inference;

-- ---------------------------------------------------------------------------
-- EXPERIMENT: one row per script execution. Holds params, cost report, and
-- aggregate eval metrics (mirrors the per-execution MLflow experiment).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference.match_experiments_0 (
    run_id                              text PRIMARY KEY,
    run_name                            text,
    model                               text NOT NULL,
    mlflow_run_id                       text,
    mlflow_experiment                   text,
    prompt_version                      text,
    seed                                integer,
    n_recipes                           integer,
    n_ingredients                       integer,
    n_llm_calls                         integer,
    n_llm_errors                        integer,
    prompt_tokens_total                 bigint,
    completion_tokens_total             bigint,
    total_tokens                        bigint,
    cost_input_usd                      double precision,
    cost_output_usd                     double precision,
    cost_total_usd                      double precision,
    abstain_rate                        double precision,
    error_rate                          double precision,
    agreement_rate                      double precision,
    staged_top1_in_llm_candidates_rate  double precision,
    staged_top1_in_top10_rate           double precision,
    certainty_mean                      double precision,
    certainty_median                    double precision,
    certainty_std                       double precision,
    certainty_p01                       double precision,
    certainty_p05                       double precision,
    certainty_p10                       double precision,
    certainty_p90                       double precision,
    elapsed_sec                         double precision,
    concurrency                         integer,
    retrieval_config                    jsonb,
    pricing                             jsonb,
    sampled_recipe_ids                  jsonb,
    status                              text,
    started_at                          timestamptz,
    finished_at                         timestamptz
);

-- ---------------------------------------------------------------------------
-- INFERENCE: one row per individual LLM call. One row per
-- (run, recipe, ingredient). Incremental checkpoints upsert here as calls land.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference.match_inferences_0 (
    inf_id                          bigserial PRIMARY KEY,
    run_id                          text NOT NULL,
    run_name                        text,
    model                           text NOT NULL,

    recipe_id                       bigint NOT NULL,
    ingredient_idx                  integer NOT NULL,
    ingredient                      text,
    name                            text,
    preparation                     text,
    dequantified                    text,
    unit                            text,

    -- LLM I/O
    system_prompt                   text,
    prompt                          text,
    response                        text,
    llm_fdc_id                      bigint,
    llm_description                 text,
    llm_certainty                   double precision,
    llm_rationale                   text,
    llm_agrees_with_staged          boolean,
    llm_abstained                   boolean,
    llm_error                       text,

    -- Staged baseline
    staged_fdc_id                   bigint,
    staged_description              text,
    staged_match_score              double precision,
    staged_match_quality            text,
    staged_base_score               double precision,
    staged_prep_score               double precision,

    -- Retrieval summary
    n_candidates_llm                integer,
    n_lexical_pool                  integer,
    n_semantic_pool                 integer,
    staged_top1_in_llm_candidates   boolean,
    staged_top1_in_top10            boolean,

    -- Directions context
    n_relevant_steps                integer,
    relevant_steps                  text,

    -- Token + cost accounting
    prompt_tokens                   integer,
    completion_tokens               integer,
    total_tokens                    integer,
    price_estimate_usd              double precision,

    -- Timestamps: single column plus split parts (per request)
    ts                              timestamptz NOT NULL,
    ts_year                         integer,
    ts_month                        integer,
    ts_date                         integer,  -- day of month
    ts_time                         text,     -- HH:MM:SS

    CONSTRAINT match_inferences_0_uniq UNIQUE (run_id, recipe_id, ingredient_idx)
);

CREATE INDEX IF NOT EXISTS match_inferences_0_run_idx
    ON inference.match_inferences_0 (run_id);
CREATE INDEX IF NOT EXISTS match_inferences_0_recipe_idx
    ON inference.match_inferences_0 (recipe_id, ingredient_idx);

-- ---------------------------------------------------------------------------
-- Top-N retrieval candidates per inference (near-miss report). Up to 10 rows
-- per (run, recipe, ingredient).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference.match_candidates_0 (
    cand_id                 bigserial PRIMARY KEY,
    run_id                  text NOT NULL,
    recipe_id               bigint NOT NULL,
    ingredient_idx          integer NOT NULL,
    rank                    integer NOT NULL,
    fdc_id                  bigint NOT NULL,
    data_type               text,
    description             text,
    lexical_dequant         double precision,
    dequant_sem             double precision,
    retrieval_score         double precision,
    staged_final_score      double precision,
    staged_base_score       double precision,
    staged_prep_score       double precision,
    in_llm_prompt           boolean,
    is_staged_top1          boolean,
    is_llm_pick             boolean,
    ts                      timestamptz NOT NULL,

    CONSTRAINT match_candidates_0_uniq UNIQUE (run_id, recipe_id, ingredient_idx, fdc_id)
);

CREATE INDEX IF NOT EXISTS match_candidates_0_run_idx
    ON inference.match_candidates_0 (run_id);
CREATE INDEX IF NOT EXISTS match_candidates_0_recipe_idx
    ON inference.match_candidates_0 (recipe_id, ingredient_idx);

-- ---------------------------------------------------------------------------
-- Spend guardrail audit log. One row per budget check performed during a run.
-- Lets the circuit breaker (and you) inspect spend decisions persistently.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference.spend_checks_0 (
    check_id                bigserial PRIMARY KEY,
    run_id                  text NOT NULL,
    check_ts                timestamptz NOT NULL,
    calls_completed         integer,
    window_start            timestamptz,         -- prior check time (rate window start)
    seconds_since_last      double precision,
    spend_since_last_usd    double precision,    -- spend in (window_start, check_ts]
    rate_usd_per_min        double precision,
    past_day_spend_usd      double precision,    -- global rolling 24h spend
    daily_limit_usd         double precision,
    rate_limit_usd_per_min  double precision,
    tripped                 boolean,
    reason                  text
);

CREATE INDEX IF NOT EXISTS spend_checks_0_run_idx
    ON inference.spend_checks_0 (run_id, check_ts);

COMMENT ON SCHEMA inference IS
    'LLM ingredient->USDA fdc_id matching; one script execution = one experiment.';
COMMENT ON TABLE inference.match_experiments_0 IS
    'One row per script execution (the experiment): params, cost report, eval metrics.';
COMMENT ON TABLE inference.match_inferences_0 IS
    'One row per LLM call: judge decision, staged baseline, tokens, and cost.';
COMMENT ON TABLE inference.match_candidates_0 IS
    'Top-N retrieval candidates (near misses) per inference for error analysis.';
COMMENT ON TABLE inference.spend_checks_0 IS
    'Budget circuit-breaker audit: per-check spend rate and daily-cap evaluation.';
