-- MVP recommendation pipeline debug logs.

CREATE SCHEMA IF NOT EXISTS mvp_logs;

CREATE TABLE IF NOT EXISTS mvp_logs.query_runs (
    run_id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          timestamptz NOT NULL DEFAULT now(),
    taste_text          text NOT NULL,
    params_json         jsonb NOT NULL,
    status              text NOT NULL,
    chosen_recipe_id    bigint,
    error_message       text
);

CREATE TABLE IF NOT EXISTS mvp_logs.stage_events (
    id                  bigserial PRIMARY KEY,
    run_id              uuid NOT NULL REFERENCES mvp_logs.query_runs(run_id) ON DELETE CASCADE,
    stage               text NOT NULL,
    seq                 integer NOT NULL,
    payload_json        jsonb NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mvp_stage_events_run_seq
    ON mvp_logs.stage_events (run_id, seq);
