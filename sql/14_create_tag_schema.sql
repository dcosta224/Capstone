-- Dietary tagging schema: nutrient dimensions, corpus percentiles, restriction flags.
-- Run after recipe schema (10) and optionally after resolved_recipes (11).

CREATE SCHEMA IF NOT EXISTS tag;

-- Dimension registry (mirrors scripts/tag_dimensions.py).
CREATE TABLE IF NOT EXISTS tag.dimension (
    id                  serial PRIMARY KEY,
    slug                text NOT NULL UNIQUE,
    nutrient_id         integer,
    unit                text NOT NULL,
    direction           text NOT NULL CHECK (direction IN ('lower_better', 'higher_better')),
    stories             text[] NOT NULL DEFAULT '{}',
    dv_per_serving      double precision,
    low_dv_frac         double precision NOT NULL DEFAULT 0.05,
    high_dv_frac        double precision NOT NULL DEFAULT 0.20,
    fallback_nutrient_id integer,
    loaded_at           timestamptz NOT NULL DEFAULT now()
);

-- Ingredient-level absolute values (per 100g from USDA).
CREATE TABLE IF NOT EXISTS tag.ingredient_nutrient (
    fdc_id              bigint NOT NULL,
    dimension_id        integer NOT NULL REFERENCES tag.dimension (id) ON DELETE CASCADE,
    absolute_per_100g   double precision,
    nutrient_id_used    integer NOT NULL,
    loaded_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fdc_id, dimension_id)
);

CREATE INDEX IF NOT EXISTS idx_ingredient_nutrient_dimension
    ON tag.ingredient_nutrient (dimension_id);

-- Recipe-level rollups (total per recipe serving = sum of ingredient contributions).
CREATE TABLE IF NOT EXISTS tag.recipe_nutrient (
    recipe_id               bigint NOT NULL,
    dimension_id            integer NOT NULL REFERENCES tag.dimension (id) ON DELETE CASCADE,
    absolute_total          double precision,
    absolute_per_serving    double precision,
    corpus_percentile       double precision,
    absolute_label          text CHECK (absolute_label IN ('low', 'medium', 'high')),
    corpus_label            text CHECK (corpus_label IN ('low', 'medium', 'high')),
    nutrient_id_used        integer,
    n_ingredients_with_value integer,
    loaded_at               timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (recipe_id, dimension_id)
);

CREATE INDEX IF NOT EXISTS idx_recipe_nutrient_dimension
    ON tag.recipe_nutrient (dimension_id);

-- Precomputed corpus percentile reference points per dimension.
CREATE TABLE IF NOT EXISTS tag.corpus_percentile (
    dimension_id        integer NOT NULL REFERENCES tag.dimension (id) ON DELETE CASCADE,
    percentile          double precision NOT NULL CHECK (percentile >= 0 AND percentile <= 100),
    value               double precision NOT NULL,
    n_recipes           integer NOT NULL,
    loaded_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (dimension_id, percentile)
);

-- Boolean restriction / allergen flags at ingredient level.
CREATE TABLE IF NOT EXISTS tag.ingredient_restriction (
    fdc_id              bigint NOT NULL,
    restriction_slug    text NOT NULL,
    source              text NOT NULL DEFAULT 'keyword',
    matched_term        text,
    loaded_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fdc_id, restriction_slug)
);

CREATE INDEX IF NOT EXISTS idx_ingredient_restriction_slug
    ON tag.ingredient_restriction (restriction_slug);

-- Recipe-level restriction rollup (any ingredient triggers = recipe contains).
CREATE TABLE IF NOT EXISTS tag.recipe_restriction (
    recipe_id           bigint NOT NULL,
    restriction_slug    text NOT NULL,
    n_triggering_lines  integer NOT NULL DEFAULT 0,
    loaded_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (recipe_id, restriction_slug)
);
