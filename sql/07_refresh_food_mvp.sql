-- Rebuild food_mvp, food_4macro, and food_4_portion_data after reloading
-- food_nutrient (all three), food_portion / measure_unit (food_mvp and food_4_portion_data only).
SET search_path TO usda, public;

REFRESH MATERIALIZED VIEW CONCURRENTLY food_mvp;

SELECT 'food_mvp' AS view_name, data_type, COUNT(*) AS foods
FROM food_mvp
GROUP BY data_type
ORDER BY foods DESC;

REFRESH MATERIALIZED VIEW CONCURRENTLY food_4macro;

SELECT 'food_4macro' AS view_name, data_type, COUNT(*) AS foods
FROM food_4macro
GROUP BY data_type
ORDER BY foods DESC;

REFRESH MATERIALIZED VIEW CONCURRENTLY food_4_portion_data;

SELECT 'food_4_portion_data' AS view_name, data_type, COUNT(*) AS foods
FROM food_4_portion_data
GROUP BY data_type
ORDER BY foods DESC;
