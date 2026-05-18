-- USDA FoodData Central (All Food, April 2026) — schema in Supabase/Postgres.
-- Idempotent: drops and recreates the usda schema.

DROP SCHEMA IF EXISTS usda CASCADE;
CREATE SCHEMA usda;
SET search_path TO usda, public;

-- Lookup / reference
CREATE TABLE food_category (
    id              integer PRIMARY KEY,
    code            text NOT NULL,
    description     text NOT NULL
);

CREATE TABLE measure_unit (
    id              integer PRIMARY KEY,
    name            text NOT NULL
);

CREATE TABLE nutrient (
    id              integer PRIMARY KEY,
    name            text NOT NULL,
    unit_name       text,
    nutrient_nbr    text,
    rank            text
);

CREATE TABLE retention_factor (
    gid             integer PRIMARY KEY,
    code            integer NOT NULL,
    food_group_id   integer NOT NULL,
    description     text NOT NULL
);

CREATE TABLE wweia_food_category (
    wweia_food_category             integer PRIMARY KEY,
    wweia_food_category_description text NOT NULL
);

CREATE TABLE lab_method (
    id              integer PRIMARY KEY,
    description     text,
    technique       text
);

-- Core food row (all data types)
CREATE TABLE food (
    fdc_id              bigint PRIMARY KEY,
    data_type           text NOT NULL,
    description         text,
    food_category_id    text,
    publication_date    date
);

-- Type-specific extensions
CREATE TABLE branded_food (
    fdc_id                          bigint PRIMARY KEY,
    brand_owner                     text,
    brand_name                      text,
    subbrand_name                   text,
    gtin_upc                        text,
    ingredients                     text,
    not_a_significant_source_of     text,
    serving_size                    double precision,
    serving_size_unit               text,
    household_serving_fulltext      text,
    branded_food_category           text,
    data_source                     text,
    package_weight                  text,
    modified_date                   date,
    available_date                  date,
    market_country                  text,
    discontinued_date               date,
    preparation_state_code          text,
    trade_channel                   text,
    short_description               text,
    material_code                   text
);

CREATE TABLE foundation_food (
    fdc_id          bigint PRIMARY KEY,
    ndb_number      text,
    footnote        text
);

CREATE TABLE sr_legacy_food (
    fdc_id          bigint PRIMARY KEY,
    ndb_number      text
);

CREATE TABLE sample_food (
    fdc_id          bigint PRIMARY KEY
);

CREATE TABLE survey_fndds_food (
    fdc_id                  bigint PRIMARY KEY,
    food_code               text,
    wweia_category_code     text,
    start_date              date,
    end_date                date
);

CREATE TABLE sub_sample_food (
    fdc_id                  bigint PRIMARY KEY,
    fdc_id_of_sample_food   bigint
);

CREATE TABLE food_component (
    id                  bigint PRIMARY KEY,
    fdc_id              bigint NOT NULL,
    name                text,
    pct_weight          text,
    is_refuse           text,
    gram_weight         double precision,
    data_points         text,
    min_year_acquired   text
);

CREATE TABLE food_portion (
    id                  bigint PRIMARY KEY,
    fdc_id              bigint NOT NULL,
    seq_num             integer,
    amount              double precision,
    measure_unit_id     integer,
    portion_description text,
    modifier            text,
    gram_weight         double precision,
    data_points         text,
    footnote            text,
    min_year_acquired   text
);

CREATE TABLE food_nutrient (
    id                      bigint PRIMARY KEY,
    fdc_id                  bigint NOT NULL,
    nutrient_id             integer NOT NULL,
    amount                  double precision,
    data_points             text,
    derivation_id           integer,
    min                     text,
    max                     text,
    median                  text,
    loq                     text,
    footnote                text,
    min_year_acquired       text,
    percent_daily_value     text
);

CREATE TABLE food_protein_conversion_factor (
    food_nutrient_conversion_factor_id  bigint PRIMARY KEY,
    value                               double precision
);

CREATE TABLE input_food (
    id                      bigint PRIMARY KEY,
    fdc_id                  bigint NOT NULL,
    fdc_id_of_input_food    text,
    seq_num                 integer,
    amount                  double precision,
    sr_code                 text,
    sr_description          text,
    unit                    text,
    portion_code            text,
    portion_description     text,
    gram_weight             double precision,
    retention_code          text
);

CREATE TABLE market_acquisition (
    fdc_id                  bigint PRIMARY KEY,
    brand_description       text,
    expiration_date         date,
    label_weight            text,
    location                text,
    acquisition_date        date,
    sales_type              text,
    sample_lot_nbr          text,
    sell_by_date            date,
    store_city              text,
    store_name              text,
    store_state             text,
    upc_code                text,
    acquisition_number      text
);

CREATE TABLE microbe (
    id              integer PRIMARY KEY,
    food_id         bigint NOT NULL,
    method          text,
    microbe_code    text,
    min_value       text,
    max_value       text,
    uom             text
);

CREATE TABLE lab_method_code (
    lab_method_id   integer NOT NULL,
    code            text NOT NULL,
    PRIMARY KEY (lab_method_id, code)
);

CREATE TABLE lab_method_nutrient (
    lab_method_id   integer NOT NULL,
    nutrient_id     integer NOT NULL,
    PRIMARY KEY (lab_method_id, nutrient_id)
);

CREATE TABLE sub_sample_result (
    food_nutrient_id    bigint NOT NULL,
    adjusted_amount     text,
    lab_method_id       integer,
    nutrient_name       text
);

CREATE TABLE fndds_ingredient_nutrient_value (
    ingredient_code             text,
    ingredient_description      text,
    nutrient_code               text,
    nutrient_value              text,
    nutrient_value_source       text,
    fdc_id                      bigint,
    derivation_code             text,
    sr_addmod_year              text,
    foundation_year_acquired    text,
    start_date                  date,
    end_date                    date
);

CREATE TABLE food_update_log_entry (
    id              bigint PRIMARY KEY,
    description     text,
    last_updated    date
);
