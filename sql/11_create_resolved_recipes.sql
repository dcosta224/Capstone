-- Per-ingredient resolution for MVP fully-resolved recipes.
-- Does not drop the recipe schema (safe to run after 10_create_recipe_schema.sql).

CREATE TABLE IF NOT EXISTS recipe.resolved_recipes (
    recipe_id             bigint NOT NULL,
    ingredient_idx        integer NOT NULL,
    recipe_name           text,
    ingredient            text NOT NULL,
    fdc_id                bigint,
    fdc_description       text,
    portion_id            bigint,
    portion_label         text,
    quantity              double precision,
    unit                  text,
    gram_weight           double precision NOT NULL,
    amount_kind           text,
    grams_status          text,
    negligible_calories   boolean NOT NULL DEFAULT false,
    feasibility_version   integer,
    loaded_at             timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (recipe_id, ingredient_idx)
);

CREATE INDEX IF NOT EXISTS idx_resolved_recipes_recipe_id
    ON recipe.resolved_recipes (recipe_id);
CREATE INDEX IF NOT EXISTS idx_resolved_recipes_fdc_id
    ON recipe.resolved_recipes (fdc_id);
CREATE INDEX IF NOT EXISTS idx_resolved_recipes_portion_id
    ON recipe.resolved_recipes (portion_id);
