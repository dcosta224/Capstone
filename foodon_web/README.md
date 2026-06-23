# FoodOn Browser

Local web app for browsing the [FoodOn](http://foodon.org) ontology hierarchy. Use it to explore how food classes are organized from broad categories (e.g. *material entity* → *food material* → *food product*) down to specific items.

## Setup

From the repo root:

```bash
# If you don't have the venv yet
uv sync

# Or with pip
python -m venv .venv
.venv/bin/pip install -e .
```

Dependencies used by the browser: `rdflib`, `fastapi`, `uvicorn` (already listed in `pyproject.toml`).

## Start the server

```bash
./scripts/run_foodon_web.sh
```

Or:

```bash
.venv/bin/python -m foodon_web.server
```

Then open **http://127.0.0.1:8765** in your browser.

### Spaghetti Carbonara composition explorer

Open **http://127.0.0.1:8765/carbonara** for an interactive view of resolved ingredient gram shares mapped to FoodOn:

- **Chart (top):** wide distribution — 1 selected node = gram-share %; 2 nodes = gram ratio
- **Tree (bottom):** tournament-style bracket, root at top, same depth = same row, labels truncated to 20 chars

### Permanent Carbonara data (no remapping on reload)

The UI reads committed JSON from `foodon_web/data/`:

- `carbonara_composition_top12.json` — full API payload (grams, hierarchy, stats)
- `carbonara_fdc_foodon_map.json` — frozen FDC→FoodOn lookups for this dish

**UI/layout changes do not trigger remapping.** Only a missing bundled file falls back to the DB + slow FoodOn search.

To regenerate after resolution data changes:

```bash
uv run python scripts/build_carbonara_foodon_cache.py
git add foodon_web/data/
```

Optional dev cache under `foodon_web/cache/` is gitignored; the bundled `data/` files are what matter.

### First run vs later runs

- **First run** downloads `foodon.owl` from OBO and builds a local index. This usually takes 10–20 seconds.
- **Later runs** load a cached index from `foodon_web/cache/foodon_index.json` and start in about a second.

The cache is gitignored; delete it if you want to force a fresh rebuild.

## Browsing the hierarchy

The tree starts at **material entity** (the BFO root used by the FoodOn browser).

### Expand a category

- Click the **▶** arrow next to a class, or click the class name itself.
- Children load on demand from the server.
- Sibling categories at the same level are hidden while one node is expanded, so you drill into one branch at a time.

### Collapse a category

- Click **▼** (or the class name again) to collapse.
- Children are hidden and any nested expansions are reset.
- Sibling categories at that level reappear.

### Counts

Each node shows a number in parentheses, e.g. `food material (20,862)`. That is the count of **unique descendant classes** below that node (same idea as the official FoodOn browser).

### BFO badge

Terms from the Basic Formal Ontology show a **BFO** pill next to the label.

### Leaf nodes

Classes with no subclasses have no arrow. Click the name to select/highlight them.

## Search

Use the search box at the top to find a class by name (fuzzy match), e.g. `romano cheese` or `tomato`.

1. Type a term and press **Enter** or click **Search**.
2. Click a result to jump to that class.
3. The app expands the path from the root down to your selection.

## Notebook

For scripted lookup and text output (no UI), see `notebooks/foodon.ipynb` — especially `print_foodon_hierarchy("your term")`.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Page won't load | Confirm the server is running and you're on port **8765**. |
| Slow startup | Normal on first run while the OWL file is downloaded and indexed. |
| Stale or broken tree | Stop the server, delete `foodon_web/cache/foodon_index.json`, restart. |
| Port already in use | Stop other processes on 8765, or change the port in `foodon_web/server.py`. |

## Project layout

```
foodon_web/
  server.py          # FastAPI app
  static/            # HTML/CSS/JS UI
  cache/             # Generated index (local only)
scripts/
  foodon_index.py    # Ontology indexing + search
  run_foodon_web.sh  # Convenience launcher
notebooks/
  foodon.ipynb       # Exploratory queries + text hierarchy
```
