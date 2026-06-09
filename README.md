# Capstone — Food & Recipe Data Pipeline

Berkeley Capstone project for loading public food-composition, recipe, and unit-conversion datasets into **Supabase Postgres**. Raw files live under `Data/` (gitignored); loaders create four logical schemas on the database.

## Repository layout

```
Capstone/
├── Data/                          # Raw datasets (not committed; see .gitignore)
│   ├── All_Food_Data_April_2026/  # USDA FoodData Central CSV export
│   ├── recipes/                   # open_recipes.json, RecipeNLG.csv
│   ├── conversions/               # food_density.csv (generated from PDF)
│   └── food_density.pdf           # FAO/INFOODS Density Database v2.0
├── scripts/                       # Python utilities and loaders
│   ├── db.py                      # Shared Supabase connection from .env
│   ├── infer_schema.py            # Introspect usda schema + infer joins
│   ├── load_recipes.py            # recipe schema (Open Recipes + RecipeNLG)
│   └── load_food_density.py       # PDF → CSV → conversions schema
├── sql/                           # DDL and psql-based USDA bulk load
│   ├── 00_create_schema.sql …     # usda tables + COPY scripts
│   ├── 10_create_recipe_schema.sql
│   ├── 20_create_conversions_schema.sql
│   └── load_*.sh                  # Shell wrappers for loaders
├── pyproject.toml                 # Project metadata and dependencies (uv)
├── uv.lock                        # Locked dependency versions
├── requirements.txt               # Pip-compatible pin list (optional)
├── .env.example                   # Supabase connection template
└── README.md
```

## Data overview

### USDA FoodData Central (`Data/All_Food_Data_April_2026/`)

April 2026 **full download** (CSV). ~25 files, ~2.1M foods in `food.csv`, ~27M rows in `food_nutrient.csv`. Loaded into the **`usda`** schema.

| Area | Main tables | Role |
|------|-------------|------|
| Core | `food`, `nutrient`, `food_category`, `measure_unit` | Every food item (`fdc_id`) and reference nutrients/units |
| Branded / legacy / FNDDS | `branded_food`, `foundation_food`, `sr_legacy_food`, `survey_fndds_food`, … | Type-specific metadata keyed by `fdc_id` |
| Composition | `food_nutrient`, `food_portion`, `food_component` | Nutrients per 100g, portions, refuse/components |
| Lab / samples | `lab_method*`, `sub_sample_*`, `market_acquisition` | Analytical methods and sample lineage |

Hub model: **`food.fdc_id`** links to extension tables (`branded_food`, `food_nutrient`, etc.). Run `uv run python scripts/infer_schema.py` for a full join map and `scripts/usda_schema_inferred.json`.

### Recipes (`Data/recipes/`)

| File | Rows (approx.) | DB table |
|------|----------------|----------|
| `open_recipes.json` | ~173k (JSON lines, schema.org-style) | `recipe.open_recipe` |
| `RecipeNLG.csv` | ~2.2M | `recipe.recipe_nlg` |

`recipe_nlg` stores `ingredients`, `directions`, and `ner` as JSON text; `open_recipe` keeps ingredients as a single text block plus URL, times, source, etc.

### Conversions (`Data/food_density.pdf` → `Data/conversions/food_density.csv`)

FAO/INFOODS **Density Database v2.0** — volume ↔ mass factors (g/ml, specific gravity). **638 foods** in **`conversions.food_density`**.

---

## Database schemas (Supabase)

| Schema | Purpose | How to load |
|--------|---------|-------------|
| `usda` | FoodData Central | `./sql/load_all.sh` (needs `psql`) |
| `recipe` | Open Recipes + RecipeNLG | `uv run python scripts/load_recipes.py` |
| `conversions` | Food density factors | `uv run python scripts/load_food_density.py` |

Connection settings are read from **`.env`** (copy from `.env.example`). Python loaders use the **session pooler** on port **5432** by default; USDA `\copy` scripts do the same.

```
postgresql://<PG_POOL_USER>:<password>@<PG_POOL_HOST>:5432/postgres?sslmode=require
```

> **Storage:** Full USDA + RecipeNLG is multi‑GB. Ensure your Supabase plan has enough disk before loading everything. RecipeNLG supports resume: `uv run python scripts/load_recipes.py --nlg-only`.

---

## Setup with [uv](https://docs.astral.sh/uv/)

[uv](https://docs.astral.sh/uv/) manages the virtualenv and dependencies via `pyproject.toml` and `uv.lock`.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

### First-time project setup

```bash
cd Capstone
cp .env.example .env   # fill in PG_PASSWORD, PG_POOL_USER, PG_POOL_HOST

uv sync                # create .venv and install locked deps
```

### Run scripts inside the project environment

Prefix commands with `uv run` so the correct interpreter and packages are used:

```bash
uv run python scripts/infer_schema.py
uv run python scripts/load_recipes.py
uv run python scripts/load_food_density.py
```

You can also activate the venv directly:

```bash
source .venv/bin/activate
python scripts/infer_schema.py
```

### Add or upgrade packages

Add a new runtime dependency (updates `pyproject.toml` and `uv.lock`):

```bash
uv add requests
```

Add a dev-only dependency:

```bash
uv add --dev pytest ruff
```

Upgrade a package to the latest compatible version:

```bash
uv add --upgrade pandas
```

Remove a package:

```bash
uv remove pandas
```

After any `uv add` / `uv remove`, commit both `pyproject.toml` and `uv.lock`. Teammates run `uv sync` to match.

### Other useful uv commands

| Command | What it does |
|---------|----------------|
| `uv sync` | Install deps from lockfile into `.venv` |
| `uv lock` | Refresh `uv.lock` after hand-editing `pyproject.toml` |
| `uv pip install -r requirements.txt` | Install from `requirements.txt` if you use that file |
| `uv run <cmd>` | Run a command in the project environment |
| `uv python pin 3.11` | Pin local Python version (see `.python-version`) |

Current project dependencies: `numpy`, `pandas`, `pdfplumber`, `psycopg2-binary`.

---

## Loading data

### 1. USDA (SQL + psql)

Requires [PostgreSQL client](https://www.postgresql.org/download/) (`psql`). From repo root:

```bash
./sql/load_all.sh
```

Runs, in order: `00_create_schema.sql` → reference/food COPY scripts → `99_create_indexes.sql`. Expect a long run for `food_nutrient.csv`.

### 2. Recipes (Python)

```bash
uv run python scripts/load_recipes.py              # full load
uv run python scripts/load_recipes.py --extract-only   # not applicable (JSON/CSV only)
uv run python scripts/load_recipes.py --nlg-only       # resume RecipeNLG after partial load
```

### 3. Food density / conversions (Python)

```bash
uv run python scripts/load_food_density.py           # PDF → CSV → DB
uv run python scripts/load_food_density.py --extract-only
uv run python scripts/load_food_density.py --load-only
```

CSV output: `Data/conversions/food_density.csv`.

### 4. Schema introspection

```bash
uv run python scripts/infer_schema.py
uv run python scripts/infer_schema.py --schema usda --out scripts/usda_schema_inferred.json
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `PG_PASSWORD` | Database password |
| `PG_POOL_USER` | Pooler user, e.g. `postgres.<project-ref>` |
| `PG_POOL_HOST` | Pooler host |
| `PG_POOL_SESSION_PORT` | Session pooler (default `5432`) |
| `PG_POOL_TRANSACTION_PORT` | Transaction pooler (`6543`) |
| `PG_DATABASE` | Database name (default `postgres`) |
| `PG_SSL_MODE` | SSL mode (default `require`) |
| `PG_PSQL_USE_TRANSACTION_POOLER_PORT` | Set to `1` to use port 6543 in Python loaders |

---

## Obtaining raw data

Place files under `Data/` (not tracked in git):

1. **USDA:** [FoodData Central download](https://fdc.nal.usda.gov/download-datasets) → “Full download of all data types” (April 2026) → unzip into `Data/All_Food_Data_April_2026/`.
2. **Recipes:** [Open-Recipes Repo](https://github.com/jakevdp/open-recipe-data/tree/main) `open_recipes.json` and [RecipeNLG Dataset](https://recipenlg.cs.put.poznan.pl/) `RecipeNLG.csv` under `Data/recipes/`.
3. **Density:** [Density PDF](https://www.fao.org/4/ap815e/ap815e.pdf) `food_density.pdf` in `Data/` (or use the copy already there).

---

## MVP: load resolved recipes + nutrients (Supabase)

After a v4 feasibility run (`scratch/EDA/portion_feasibility_1000_v4_no_portion/pipeline_matches.parquet`), load the **106 fully-resolved recipes** into Supabase:

```bash
# 1. Per-ingredient fdc_id + gram_weight (+ portion_id where applicable)
uv run python scripts/load_resolved_recipes.py --dry-run
uv run python scripts/load_resolved_recipes.py --execute

# 2. Per-recipe nutrient totals (wide table; amounts scaled from per-100g USDA values)
uv run python scripts/load_recipe_nutrients.py --dry-run
uv run python scripts/load_recipe_nutrients.py --execute
```

| Table | Rows (expected) | Contents |
|-------|-----------------|----------|
| `recipe.resolved_recipes` | ~634 | One row per ingredient line |
| `recipe.recipe_nutrients` | 106 | One row per recipe; columns like `energy_kcal`, `protein_g`, … |

DDL: [`sql/11_create_resolved_recipes.sql`](sql/11_create_resolved_recipes.sql). Nutrient table schema is generated from `usda.nutrient` at load time.

EDA: [`scratch/EDA/portion_feasibility_v4_recipe_eda.ipynb`](scratch/EDA/portion_feasibility_v4_recipe_eda.ipynb)

---

## MVP recipe recommendation (local web app)

Ranks the **106 fully-resolved recipes** by semantic similarity (MiniLM) and PFC calorie-fraction fit, optimizes portions on the top 10 via convex SCA (PFC gram targets at the **midpoint kcal** of your calorie range, with total kcal fixed at that midpoint), then uses an LLM judge for the final pick. The UI streams pipeline progress in real time over SSE.

### Prerequisites

- Supabase tables populated: `recipe.resolved_recipes`, `recipe.recipe_nutrients`, `recipe.recipe_nlg_features`, `recipe.recipe_nlg_embedding`
- `.env` with Postgres pool vars (see above)
- `OPENAI_API_KEY` for LLM judge (falls back to deterministic mock if unset)
- `HF_TOKEN` optional (Hugging Face model download cache)

Apply debug-log schema (optional, also auto-created on first run):

```bash
psql "$DATABASE_URL" -f sql/13_create_mvp_logs_schema.sql
```

### Run locally

```bash
uv sync
# Optional: pre-build disk cache (faster first request; uses local embeddings export if present)
uv run python scripts/warm_mvp_cache.py

uv run uvicorn mvp_web.server:app --reload --port 8000
```

On startup the server **warms the in-memory cache** (106 recipes, embeddings, ingredients, USDA nutrients) from `mvp_web/cache/mvp_corpus.pkl` or Supabase. Subsequent demo queries skip corpus DB loads.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Submit taste text plus calorie and PFC % ranges. The UI streams pipeline stages (ranking → optimizer → judge) and shows the structured recipe with USDA descriptions, portion labels, and optimized quantities.

### Tests

```bash
uv run pytest tests/ -v
```

### Key modules

| Path | Role |
|------|------|
| `scripts/recipe_macro_optimizer.py` | Log-ratio portion optimizer (CVXPY + SCA) |
| `scripts/mvp_nutrient_fit.py` | PFC calorie-fraction range distance |
| `scripts/mvp_recipe_ranker.py` | Stage-1 combined ranking |
| `scripts/mvp_pipeline.py` | Full orchestrator + SSE events |
| `scripts/mvp_recipe_judge.py` | LLM final selection |
| `mvp_web/server.py` | FastAPI + SSE endpoint |

Pipeline runs are logged to `mvp_logs.query_runs` and `mvp_logs.stage_events` when Supabase is reachable.

---

## License & attribution

- USDA FoodData Central — U.S. Department of Agriculture  
- Open Recipes / RecipeNLG — see original dataset terms  
- FAO/INFOODS Density Database v2.0 — FAO/INFOODS
