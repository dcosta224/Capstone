-- Vector + lookup indexes for recipe embeddings. Run after load_recipe_embeddings.py.
SET search_path TO recipe, public;

CREATE INDEX IF NOT EXISTS idx_recipe_nlg_features_semantic_text
    ON recipe_nlg_features (semantic_text);

-- HNSW supports incremental inserts on Supabase pgvector; build after initial load.
CREATE INDEX IF NOT EXISTS idx_recipe_nlg_embedding_hnsw
    ON recipe_nlg_embedding
    USING hnsw (embedding vector_cosine_ops);

ANALYZE recipe_nlg_features;
ANALYZE recipe_nlg_embedding;
