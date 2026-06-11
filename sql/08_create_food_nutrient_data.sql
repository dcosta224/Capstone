-- Materialized view: all food rows with at least one food_nutrient row.
-- Run after food and food_nutrient are loaded (sql/load_all.sh).
-- Refresh when food_nutrient changes:
--   REFRESH MATERIALIZED VIEW CONCURRENTLY usda.food_nutrient_data;

SET search_path TO usda, public;

DROP MATERIALIZED VIEW IF EXISTS food_nutrient_data;

CREATE MATERIALIZED VIEW food_nutrient_data AS
SELECT f.*
FROM food f
WHERE EXISTS (
    SELECT 1
    FROM food_nutrient fn
    WHERE fn.fdc_id = f.fdc_id
);

CREATE UNIQUE INDEX food_nutrient_data_fdc_id_idx ON food_nutrient_data (fdc_id);

COMMENT ON MATERIALIZED VIEW food_nutrient_data IS
    'All usda.food rows that have at least one food_nutrient row (any data_type).';
