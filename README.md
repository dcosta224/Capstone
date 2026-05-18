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

## License & attribution

- USDA FoodData Central — U.S. Department of Agriculture  
- Open Recipes / RecipeNLG — see original dataset terms  
- FAO/INFOODS Density Database v2.0 — FAO/INFOODS
