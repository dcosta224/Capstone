#!/usr/bin/env bash
# Load USDA April 2026 CSVs into Supabase Postgres (usda schema).
#
# Connection (same shape as former scratch.py):
#   postgresql://postgres.<project-ref>:<password>@<pool-host>:<port>/postgres?sslmode=require
#
# Credentials from repo .env (PG_POOL_*, PG_PASSWORD, PG_SSL_MODE).
# Uses session pooler port 5432 by default (required for \copy / long COPY).
# Set PG_PSQL_USE_TRANSACTION_POOLER_PORT=1 to use port 6543 instead.

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

echo "Target: ${PG_POOL_USER}@${PG_POOL_HOST}:${PG_PORT}/${PG_DATABASE} (schema: usda)"
echo "Data:   ${ROOT}/Data/All_Food_Data_April_2026/"

run_sql() {
  local file="$1"
  echo "==> ${file}"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$file"
}

run_sql sql/00_create_schema.sql
run_sql sql/01_load_reference.sql
run_sql sql/02_load_food.sql
run_sql sql/03_load_food_types.sql
run_sql sql/04_load_food_details.sql
run_sql sql/05_load_lab_and_fndds.sql
run_sql sql/99_create_indexes.sql

echo "Done. Row counts:"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT relname AS table_name, n_live_tup AS approx_rows
FROM pg_stat_user_tables
WHERE schemaname = 'usda'
ORDER BY relname;
"
