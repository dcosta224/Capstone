-- FNDDS Ingredient Nutrient Values (IngredNutVal) — standalone load for Supabase.
--
-- Source CSV (April 2026 All Food bundle):
--   Data/All_Food_Data_April_2026/fndds_ingredient_nutrient_value.csv
--
-- CSV headers (positional; names differ from DB columns):
--   "ingredient code", "Ingredient description", "Nutrient code", "Nutrient value",
--   "Nutrient value source", "FDC ID", "Derivation code", "SR AddMod year",
--   "Foundation year acquired", "Start date", "End date"
--
-- Load from project root with psql (uses session pooler port from .env):
--
--   export PGPASSWORD="$PG_PASSWORD"
--   psql "postgresql://${PG_POOL_USER}@${PG_POOL_HOST}:${PG_POOL_SESSION_PORT:-5432}/${PG_DATABASE:-postgres}?sslmode=${PG_SSL_MODE:-require}" \
--     -v ON_ERROR_STOP=1 \
--     -f sql/33_create_fndds_ingredient_nutrient_value.sql
--
-- Or create the table in Supabase SQL Editor (through CREATE TABLE below), then either:
--   - Table Editor → Import CSV (map columns to snake_case names below), or
--   - run only the \copy block via psql from your machine (Supabase SQL Editor cannot read local files).
--
-- Idempotent: drops and recreates this table only (does not touch other usda tables).

\set ON_ERROR_STOP on
SET search_path TO usda, public;

CREATE SCHEMA IF NOT EXISTS usda;

DROP TABLE IF EXISTS usda.fndds_ingredient_nutrient_value;

CREATE TABLE usda.fndds_ingredient_nutrient_value (
    ingredient_code             text NOT NULL,
    ingredient_description      text,
    nutrient_code               text NOT NULL,
    nutrient_value              double precision,
    nutrient_value_source       text,
    fdc_id                      bigint,
    derivation_code             text,
    sr_addmod_year              text,
    foundation_year_acquired      text,
    start_date                  date,
    end_date                    date
);

COMMENT ON TABLE usda.fndds_ingredient_nutrient_value IS
    'FNDDS IngredNutVal: nutrient amounts per 100 g edible portion for FNDDS ingredient codes (NDB numbers).';

-- ~275k rows; COPY is faster than INSERT from the dashboard for full reloads.
\copy usda.fndds_ingredient_nutrient_value (
    ingredient_code,
    ingredient_description,
    nutrient_code,
    nutrient_value,
    nutrient_value_source,
    fdc_id,
    derivation_code,
    sr_addmod_year,
    foundation_year_acquired,
    start_date,
    end_date
) FROM 'Data/All_Food_Data_April_2026/fndds_ingredient_nutrient_value.csv'
  WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"', NULL '');

CREATE INDEX IF NOT EXISTS idx_fndds_ingredient_code
    ON usda.fndds_ingredient_nutrient_value (ingredient_code);

CREATE INDEX IF NOT EXISTS idx_fndds_ingredient_nutrient_code
    ON usda.fndds_ingredient_nutrient_value (ingredient_code, nutrient_code);

CREATE INDEX IF NOT EXISTS idx_fndds_ingredient_fdc_id
    ON usda.fndds_ingredient_nutrient_value (fdc_id);

ANALYZE usda.fndds_ingredient_nutrient_value;
