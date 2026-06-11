-- Add low-tail certainty percentiles to experiment summary rows.
-- Idempotent: safe to re-run.

ALTER TABLE inference.match_experiments_0
    ADD COLUMN IF NOT EXISTS certainty_p01 double precision,
    ADD COLUMN IF NOT EXISTS certainty_p05 double precision;
