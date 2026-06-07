-- Rebuild food_nutrient_data after reloading food or food_nutrient.
SET search_path TO usda, public;

REFRESH MATERIALIZED VIEW CONCURRENTLY food_nutrient_data;

SELECT data_type, COUNT(*) AS foods
FROM food_nutrient_data
GROUP BY data_type
ORDER BY foods DESC;
