"""Local web UI for browsing the FoodOn ontology hierarchy."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from carbonara_data import (  # noqa: E402
    CARBONARA_RECIPE_ID,
    DEFAULT_TOP_N,
    build_composition_payload,
    fetch_canonical_lines,
    load_bundled_composition,
    load_composition_cache,
    load_fdc_map_cache,
    save_composition_cache,
)
from db import connect, load_dotenv  # noqa: E402
from foodon_index import FoodOnIndex  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "foodon_index.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
COMPOSITION_CACHE: dict[tuple[int, int], dict] = {}

_index: FoodOnIndex | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _index
    load_fdc_map_cache()
    _index = FoodOnIndex.from_owl(cache_path=CACHE_PATH)
    yield
    _index = None


app = FastAPI(title="FoodOn Browser", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _require_index() -> FoodOnIndex:
    if _index is None:
        raise HTTPException(status_code=503, detail="FoodOn index is still loading")
    return _index


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/carbonara")
def carbonara_page():
    return FileResponse(STATIC_DIR / "carbonara.html")


def _canonical_title(conn, canonical_recipe_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT title FROM recipe.canonical_recipes WHERE id = %s",
            (canonical_recipe_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown canonical recipe: {canonical_recipe_id}")
    return row[0]


def _composition_payload(canonical_recipe_id: int, top_n: int) -> dict:
    cache_key = (canonical_recipe_id, top_n)
    if cache_key in COMPOSITION_CACHE:
        return COMPOSITION_CACHE[cache_key]

    bundled = load_bundled_composition(canonical_recipe_id, top_n)
    if bundled is not None:
        COMPOSITION_CACHE[cache_key] = bundled
        return bundled

    cached = load_composition_cache(canonical_recipe_id, top_n)
    if cached is not None:
        COMPOSITION_CACHE[cache_key] = cached
        return cached

    load_dotenv()
    index = _require_index()
    conn = connect()
    try:
        title = _canonical_title(conn, canonical_recipe_id)
        lines_df = fetch_canonical_lines(conn, canonical_recipe_id)
    finally:
        conn.close()

    payload = build_composition_payload(
        index,
        lines_df,
        canonical_recipe_id=canonical_recipe_id,
        title=title,
        top_n=top_n,
    )
    save_composition_cache(payload, canonical_recipe_id, top_n)
    COMPOSITION_CACHE[cache_key] = payload
    return payload


@app.get("/api/canonical-recipes/{canonical_recipe_id}/foodon-composition")
def foodon_composition(
    canonical_recipe_id: int,
    top_n: int = Query(default=DEFAULT_TOP_N, ge=1, le=30),
):
    return _composition_payload(canonical_recipe_id, top_n)


@app.get("/api/carbonara/foodon-composition")
def carbonara_composition(top_n: int = Query(default=DEFAULT_TOP_N, ge=1, le=30)):
    return _composition_payload(CARBONARA_RECIPE_ID, top_n)


@app.get("/api/status")
def status():
    index = _require_index()
    return {
        "class_count": len(index.labels),
        "root_count": len(index.roots),
        "cache_path": str(CACHE_PATH),
        "cache_exists": CACHE_PATH.exists(),
    }


@app.get("/api/roots")
def roots():
    index = _require_index()
    return {
        "nodes": [index.node_summary(node_id) for node_id in index.preferred_roots()],
    }


@app.get("/api/nodes/{node_id}")
def node_detail(node_id: str):
    index = _require_index()
    summary = index.node_summary(node_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    return summary


@app.get("/api/nodes/{node_id}/children")
def node_children(node_id: str):
    index = _require_index()
    if node_id not in index.labels:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    return {
        "id": node_id,
        "label": index.labels[node_id],
        "children": index.child_summaries(node_id),
    }


@app.get("/api/nodes/{node_id}/ancestors")
def node_ancestors(node_id: str):
    index = _require_index()
    if node_id not in index.labels:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    path = index.ancestry_path(node_id)
    return {
        "id": node_id,
        "ancestors": [index.node_summary(ancestor_id) for ancestor_id in path],
    }


@app.get("/api/search")
def search(
    q: str = Query(min_length=1),
    limit: int = Query(default=25, ge=1, le=100),
):
    index = _require_index()
    return {"query": q, "results": index.search(q, limit=limit)}


def main():
    import uvicorn

    uvicorn.run("foodon_web.server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
