-- Materialized view: non-branded foods with both nutrient and portion data.
-- Run after loading base tables (sql/load_all.sh). Refresh when underlying data changes:
--   REFRESH MATERIALIZED VIEW CONCURRENTLY usda.food_mvp;

SET search_path TO usda, public;

DROP MATERIALIZED VIEW IF EXISTS food_mvp;

CREATE MATERIALIZED VIEW food_mvp AS
SELECT
    f.fdc_id,
    f.data_type,
    f.description,
    f.food_category_id,
    f.publication_date
FROM food f
WHERE f.data_type <> 'branded_food'
  AND EXISTS (
      SELECT 1
      FROM food_nutrient fn
      WHERE fn.fdc_id = f.fdc_id
  )
  AND EXISTS (
      SELECT 1
      FROM food_portion fp
      WHERE fp.fdc_id = f.fdc_id
  );

CREATE UNIQUE INDEX food_mvp_fdc_id_idx ON food_mvp (fdc_id);

COMMENT ON MATERIALIZED VIEW food_mvp IS
    'Non-branded foods (fdc_id) that have at least one row in food_nutrient and food_portion.';
