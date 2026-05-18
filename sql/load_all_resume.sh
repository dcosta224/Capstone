#!/usr/bin/env bash
# Resume USDA load after a partial run (does not drop schema).
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
: "${PG_POOL_USER:?}"
: "${PG_POOL_HOST:?}"

PG_DATABASE="${PG_DATABASE:-postgres}"
PG_SSL_MODE="${PG_SSL_MODE:-require}"
PG_PORT="${PG_POOL_SESSION_PORT:-5432}"
export PGPASSWORD="$PG_PASSWORD"
export PGOPTIONS="${PGOPTIONS:--c statement_timeout=0}"
DATABASE_URL="postgresql://${PG_POOL_USER}@${PG_POOL_HOST}:${PG_PORT}/${PG_DATABASE}?sslmode=${PG_SSL_MODE}"

echo "Fixing columns that allow empty CSV fields …"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<'SQL'
ALTER TABLE usda.branded_food
  ALTER COLUMN serving_size TYPE text USING serving_size::text,
  ALTER COLUMN modified_date TYPE text USING modified_date::text,
  ALTER COLUMN available_date TYPE text USING available_date::text,
  ALTER COLUMN discontinued_date TYPE text USING discontinued_date::text;
ALTER TABLE usda.food
  ALTER COLUMN publication_date TYPE text USING publication_date::text;
ALTER TABLE usda.survey_fndds_food
  ALTER COLUMN start_date TYPE text USING start_date::text,
  ALTER COLUMN end_date TYPE text USING end_date::text;
ALTER TABLE usda.market_acquisition
  ALTER COLUMN expiration_date TYPE text USING expiration_date::text,
  ALTER COLUMN acquisition_date TYPE text USING acquisition_date::text,
  ALTER COLUMN sell_by_date TYPE text USING sell_by_date::text;
ALTER TABLE usda.fndds_ingredient_nutrient_value
  ALTER COLUMN start_date TYPE text USING start_date::text,
  ALTER COLUMN end_date TYPE text USING end_date::text;
ALTER TABLE usda.food_update_log_entry
  ALTER COLUMN last_updated TYPE text USING last_updated::text;
SQL

for f in sql/03_load_food_types.sql sql/04_load_food_details.sql sql/05_load_lab_and_fndds.sql sql/99_create_indexes.sql; do
  echo "==> $f"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"
done

psql "$DATABASE_URL" -c "
SELECT relname, n_live_tup FROM pg_stat_user_tables
WHERE schemaname = 'usda' ORDER BY n_live_tup DESC LIMIT 15;
"
