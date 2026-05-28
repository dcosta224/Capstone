-- Materialized views for foods usable in recipe/nutrition pipelines.
--
-- text_has_volume: shared volume-unit detector (keep in sync with scripts/usda_volume_units.py).
--
-- food_mvp: non-branded, any food_nutrient row, density-inferable portion (~7,700 full CSV load).
--
-- food_4macro: same as food_4_portion_data but without the density-inferable portion
--   requirement (four core macros only).
--
-- food_4_portion_data: all data_type values with all four core macros
--   (nutrient_id 1003 protein, 1004 fat, 1005 carbohydrate, 1008 energy)
--   and at least one density-inferable portion (gram_weight > 0 + volume unit).
--   One row per normalize_food_description(description); ties: foundation_food first,
--   then latest publication_date (see scripts/ingredient_match.normalize_text).
--
-- Requires complete food_nutrient (~28M rows); partial loads yield fewer rows.
-- Run after loading base tables (sql/load_all.sh). Refresh:
--   psql ... -f sql/07_refresh_food_mvp.sql

SET search_path TO usda, public;

-- Volume-unit detection (keep in sync with scripts/usda_volume_units.VOLUME_PATTERN).
CREATE OR REPLACE FUNCTION text_has_volume(
    modifier text,
    portion_description text,
    measure_unit_name text
) RETURNS boolean
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
SELECT (
    COALESCE(modifier, '') || ' ' ||
    COALESCE(portion_description, '') || ' ' ||
    COALESCE(measure_unit_name, '')
) ~* (
    '\m('
    'cups?|'
    'tsp|teaspoon|teaspoons|'
    'tbsp|tablespoon|tablespoons|'
    'fl\.?\s*oz|fluid\s*ounce|'
    'liter|litre|liters|litres|'
    'ml|milliliter|milliliters|'
    'pint|pints|quart|quarts|gallon|gallons|'
    'cubic\s*inch|cubic\s*inches|cubic\s*centimeter|cubic\s*centimeters|cubic\s*cm|'
    'cc'
    ')\M'
);
$$;

-- Match scripts/ingredient_match.normalize_text (casefold, strip punctuation, collapse spaces).
CREATE OR REPLACE FUNCTION normalize_food_description(description text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
SELECT trim(
    regexp_replace(
        regexp_replace(
            lower(trim(COALESCE(description, ''))),
            '[^\w\s]+',
            ' ',
            'g'
        ),
        '\s+',
        ' ',
        'g'
    )
);
$$;

-- ---------------------------------------------------------------------------
-- food_mvp
-- ---------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS food_mvp;

CREATE MATERIALIZED VIEW food_mvp AS
SELECT f.*
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
      LEFT JOIN measure_unit mu ON mu.id = fp.measure_unit_id
      WHERE fp.fdc_id = f.fdc_id
        AND fp.gram_weight > 0
        AND text_has_volume(fp.modifier, fp.portion_description, mu.name)
  );

CREATE UNIQUE INDEX food_mvp_fdc_id_idx ON food_mvp (fdc_id);

COMMENT ON MATERIALIZED VIEW food_mvp IS
    'Non-branded foods with nutrients and at least one density-inferable portion '
    '(gram_weight > 0 + volume unit in modifier, portion_description, or measure_unit).';

-- ---------------------------------------------------------------------------
-- food_4_portion_data
-- ---------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS food_4_portion_data;

CREATE MATERIALIZED VIEW food_4_portion_data AS
WITH macro_pts AS (
    SELECT
        fdc_id,
        COUNT(DISTINCT nutrient_id) AS n_macro_pts
    FROM food_nutrient
    WHERE nutrient_id IN (1003, 1004, 1005, 1008)  -- protein, fat, carb, energy
    GROUP BY fdc_id
),
eligible AS (
    SELECT f.*
    FROM food f
    INNER JOIN macro_pts mp ON f.fdc_id = mp.fdc_id
    WHERE mp.n_macro_pts = 4
      AND EXISTS (
          SELECT 1
          FROM food_portion fp
          LEFT JOIN measure_unit mu ON mu.id = fp.measure_unit_id
          WHERE fp.fdc_id = f.fdc_id
            AND fp.gram_weight > 0
            AND text_has_volume(fp.modifier, fp.portion_description, mu.name)
      )
),
deduped AS (
    SELECT
        e.*,
        ROW_NUMBER() OVER (
            PARTITION BY normalize_food_description(e.description)
            ORDER BY
                CASE WHEN e.data_type = 'foundation_food' THEN 0 ELSE 1 END,
                e.publication_date DESC NULLS LAST,
                e.fdc_id DESC
        ) AS dedupe_rank
    FROM eligible e
    WHERE normalize_food_description(e.description) <> ''
)
SELECT
    fdc_id,
    data_type,
    description,
    food_category_id,
    publication_date
FROM deduped
WHERE dedupe_rank = 1;

CREATE UNIQUE INDEX food_4_portion_data_fdc_id_idx ON food_4_portion_data (fdc_id);

COMMENT ON MATERIALIZED VIEW food_4_portion_data IS
    'Foods with protein (1003), fat (1004), carbohydrate (1005), and energy (1008), '
    'at least one density-inferable portion, deduped by normalize_food_description. '
    'Canonical row: foundation_food when present in a duplicate group, else latest publication_date.';

-- ---------------------------------------------------------------------------
-- food_4macro
-- ---------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS food_4macro;

CREATE MATERIALIZED VIEW food_4macro AS
WITH macro_pts AS (
    SELECT
        fdc_id,
        COUNT(DISTINCT nutrient_id) AS n_macro_pts
    FROM food_nutrient
    WHERE nutrient_id IN (1003, 1004, 1005, 1008)  -- protein, fat, carb, energy
    GROUP BY fdc_id
),
eligible AS (
    SELECT f.*
    FROM food f
    INNER JOIN macro_pts mp ON f.fdc_id = mp.fdc_id
    WHERE mp.n_macro_pts = 4
),
deduped AS (
    SELECT
        e.*,
        ROW_NUMBER() OVER (
            PARTITION BY normalize_food_description(e.description)
            ORDER BY
                CASE WHEN e.data_type = 'foundation_food' THEN 0 ELSE 1 END,
                e.publication_date DESC NULLS LAST,
                e.fdc_id DESC
        ) AS dedupe_rank
    FROM eligible e
    WHERE normalize_food_description(e.description) <> ''
)
SELECT
    fdc_id,
    data_type,
    description,
    food_category_id,
    publication_date
FROM deduped
WHERE dedupe_rank = 1;

CREATE UNIQUE INDEX food_4macro_fdc_id_idx ON food_4macro (fdc_id);

COMMENT ON MATERIALIZED VIEW food_4macro IS
    'Foods with protein (1003), fat (1004), carbohydrate (1005), and energy (1008), '
    'deduped by normalize_food_description (no portion requirement). '
    'Same canonical row rule as food_4_portion_data.';

COMMENT ON FUNCTION normalize_food_description(text) IS
    'Normalized food description for dedup/matching (see ingredient_match.normalize_text).';

COMMENT ON FUNCTION text_has_volume(text, text, text) IS
    'True when combined portion text matches volume units (see usda_volume_units.py).';
