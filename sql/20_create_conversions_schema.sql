-- FAO/INFOODS Density Database v2.0 (food_density.pdf)

DROP SCHEMA IF EXISTS conversions CASCADE;
CREATE SCHEMA conversions;

CREATE TABLE conversions.food_density (
    id                  serial PRIMARY KEY,
    food_group          text,
    food_name           text NOT NULL,
    density_g_per_ml    text NOT NULL,
    specific_gravity    text,
    biblio_id           text,
    updated_version_2   boolean NOT NULL DEFAULT false
);

CREATE INDEX idx_food_density_food_group ON conversions.food_density (food_group);
CREATE INDEX idx_food_density_food_name ON conversions.food_density (food_name);
