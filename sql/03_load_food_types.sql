-- Data-type extension tables (depend on food.fdc_id logically; no FK enforced for load speed)
\set ON_ERROR_STOP on
SET search_path TO usda, public;

\copy branded_food FROM 'Data/All_Food_Data_April_2026/branded_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy foundation_food (fdc_id, ndb_number, footnote) FROM 'Data/All_Food_Data_April_2026/foundation_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy sr_legacy_food (fdc_id, ndb_number) FROM 'Data/All_Food_Data_April_2026/sr_legacy_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy sample_food FROM 'Data/All_Food_Data_April_2026/sample_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy survey_fndds_food FROM 'Data/All_Food_Data_April_2026/survey_fndds_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
\copy sub_sample_food FROM 'Data/All_Food_Data_April_2026/sub_sample_food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
