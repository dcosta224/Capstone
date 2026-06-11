#!/usr/bin/env bash
# Apply recipe embedding DDL and load features + vectors from exploration.ipynb logic.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${PG_PASSWORD:?Set PG_PASSWORD in .env}"
: "${PG_POOL_USER:?Set PG_POOL_USER in .env}"
: "${PG_POOL_HOST:?Set PG_POOL_HOST in .env}"

PG_DATABASE="${PG_DATABASE:-postgres}"
PG_SSL_MODE="${PG_SSL_MODE:-require}"

if [[ "${PG_PSQL_USE_TRANSACTION_POOLER_PORT:-0}" == "1" ]]; then
  PG_PORT="${PG_POOL_TRANSACTION_PORT:-6543}"
else
  PG_PORT="${PG_POOL_SESSION_PORT:-5432}"
fi

export PGPASSWORD="$PG_PASSWORD"
export PGOPTIONS="${PGOPTIONS:--c statement_timeout=0}"
DATABASE_URL="postgresql://${PG_POOL_USER}@${PG_POOL_HOST}:${PG_PORT}/${PG_DATABASE}?sslmode=${PG_SSL_MODE}"

echo "==> sql/11_create_recipe_embedding_schema.sql"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/11_create_recipe_embedding_schema.sql

echo "==> scripts/load_recipe_embeddings.py"
uv run python scripts/load_recipe_embeddings.py "$@"

echo "==> sql/12_create_recipe_embedding_indexes.sql (optional; may take a while)"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/12_create_recipe_embedding_indexes.sql
