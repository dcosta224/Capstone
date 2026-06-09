-- Recipe text embeddings (Kadin exploration.ipynb semantic_text + MiniLM).
-- Lives in the existing `recipe` schema beside recipe_nlg.
-- Requires: recipe.recipe_nlg populated, pgvector enabled on Supabase.
--
-- Load:
--   uv run python scripts/load_recipe_embeddings.py

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS recipe.recipe_nlg_features (
    recipe_id           bigint PRIMARY KEY
        REFERENCES recipe.recipe_nlg (id) ON DELETE CASCADE,
    title_clean         text NOT NULL,
    semantic_text       text NOT NULL,
    ingredient_count    integer NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE recipe.recipe_nlg_features IS
    'Cleaned title + sorted ingredient tokens used to build semantic_text for embedding.';

CREATE TABLE IF NOT EXISTS recipe.recipe_nlg_embedding (
    recipe_id           bigint PRIMARY KEY
        REFERENCES recipe.recipe_nlg (id) ON DELETE CASCADE,
    model               text NOT NULL,
    dims                smallint NOT NULL,
    embedding           vector(384) NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE recipe.recipe_nlg_embedding IS
    'L2-normalized sentence-transformers/all-MiniLM-L6-v2 vectors over semantic_text.';
