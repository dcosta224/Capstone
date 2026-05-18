-- Lab crosswalks, FNDDS ingredient values, update log
\set ON_ERROR_STOP on
SET search_path TO usda, public;

\copy lab_method_code FROM 'Data/All_Food_Data_April_2026/lab_method_code.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy lab_method_nutrient FROM 'Data/All_Food_Data_April_2026/lab_method_nutrient.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy sub_sample_result FROM 'Data/All_Food_Data_April_2026/sub_sample_result.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy fndds_ingredient_nutrient_value (
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
) FROM 'Data/All_Food_Data_April_2026/fndds_ingredient_nutrient_value.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy food_update_log_entry FROM 'Data/All_Food_Data_April_2026/food_update_log_entry.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
