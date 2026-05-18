-- Recipe datasets: Open Recipes (JSON) and RecipeNLG (CSV)

DROP SCHEMA IF EXISTS recipe CASCADE;
CREATE SCHEMA recipe;

CREATE TABLE recipe.open_recipe (
    id                  text PRIMARY KEY,
    name                text,
    ingredients         text,
    url                 text,
    image               text,
    source              text,
    description         text,
    creator             text,
    recipe_yield        text,
    recipe_category     text,
    cook_time           text,
    prep_time           text,
    total_time          text,
    date_published      text,
    date_modified       text,
    ts_millis           bigint
);

CREATE TABLE recipe.recipe_nlg (
    id                  bigint PRIMARY KEY,
    title               text NOT NULL,
    ingredients         jsonb NOT NULL,
    directions          jsonb NOT NULL,
    link                text,
    source              text,
    ner                 jsonb NOT NULL
);

CREATE INDEX idx_open_recipe_source ON recipe.open_recipe (source);
CREATE INDEX idx_recipe_nlg_source ON recipe.recipe_nlg (source);
CREATE INDEX idx_recipe_nlg_title ON recipe.recipe_nlg (title);
