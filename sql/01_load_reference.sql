-- Load lookup tables (run from repo root: psql ... -f sql/01_load_reference.sql)
\set ON_ERROR_STOP on
SET search_path TO usda, public;

\copy food_category FROM 'Data/All_Food_Data_April_2026/food_category.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy measure_unit FROM 'Data/All_Food_Data_April_2026/measure_unit.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy nutrient (id, name, unit_name, nutrient_nbr, rank) FROM 'Data/All_Food_Data_April_2026/nutrient.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy retention_factor (gid, code, food_group_id, description) FROM 'Data/All_Food_Data_April_2026/retention_factor.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy wweia_food_category FROM 'Data/All_Food_Data_April_2026/wweia_food_category.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy lab_method FROM 'Data/All_Food_Data_April_2026/lab_method.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
