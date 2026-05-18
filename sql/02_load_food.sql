-- Core food table (~2.1M rows)
\set ON_ERROR_STOP on
SET search_path TO usda, public;

\copy food FROM 'Data/All_Food_Data_April_2026/food.csv' WITH (FORMAT csv, HEADER true, QUOTE '"', ESCAPE '"')
