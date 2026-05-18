-- Nutrient amounts, portions, components, acquisitions (~28M rows in food_nutrient)
\set ON_ERROR_STOP on
SET search_path TO usda, public;

\copy food_component FROM 'Data/All_Food_Data_April_2026/food_component.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy food_portion FROM 'Data/All_Food_Data_April_2026/food_portion.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy food_nutrient FROM 'Data/All_Food_Data_April_2026/food_nutrient.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy food_protein_conversion_factor FROM 'Data/All_Food_Data_April_2026/food_protein_conversion_factor.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy input_food FROM 'Data/All_Food_Data_April_2026/input_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy market_acquisition FROM 'Data/All_Food_Data_April_2026/market_acquisition.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy microbe (id, food_id, method, microbe_code, min_value, max_value, uom) FROM 'Data/All_Food_Data_April_2026/microbe.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
