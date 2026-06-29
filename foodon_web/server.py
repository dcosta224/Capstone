"""Local web UI for browsing the FoodOn ontology hierarchy."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_contains_core import load_contains_table  # noqa: E402
from foodon_index import FoodOnIndex  # noqa: E402
from foodon_paths import FOODON_CONTAINS_SUMMARY, FOODON_INDEX_CACHE  # noqa: E402

try:
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

    CARBONARA_AVAILABLE = True
except ImportError:
    CARBONARA_AVAILABLE = False
    CARBONARA_RECIPE_ID = 0
    DEFAULT_TOP_N = 12

WEB_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "foodon_index.json"
CACHE_PATH = WEB_CACHE_PATH if WEB_CACHE_PATH.is_file() else FOODON_INDEX_CACHE
STATIC_DIR = Path(__file__).resolve().parent / "static"
COMPOSITION_CACHE: dict[tuple[int, int], dict] = {}

_index: FoodOnIndex | None = None
_contains_active: dict[str, list[str]] = {}
_contains_flags: dict[str, dict[str, bool]] = {}
_contains_slugs: list[str] = []
_contains_summary: dict[str, Any] = {}


def _init_contains() -> None:
    global _contains_slugs, _contains_summary
    table = load_contains_table()
    if table is None:
        return

    _contains_slugs[:] = [
        col.replace("contains_", "")
        for col in table.columns
        if str(col).startswith("contains_")
    ]

    for _, row in table.iterrows():
        node_id = str(row["foodon_id"])
        flags = {slug: bool(row.get(f"contains_{slug}", False)) for slug in _contains_slugs}
        _contains_flags[node_id] = flags
        active = [slug for slug, val in flags.items() if val]
        if active:
            _contains_active[node_id] = active

    if FOODON_CONTAINS_SUMMARY.is_file():
        import json

        _contains_summary.update(json.loads(FOODON_CONTAINS_SUMMARY.read_text(encoding="utf-8")))
    else:
        _contains_summary.update(
            {
                "nodes": len(table),
                "contains_slugs": _contains_slugs,
                "tagged_counts": {
                    slug: int(table[f"contains_{slug}"].sum())
                    for slug in _contains_slugs
                    if f"contains_{slug}" in table.columns
                },
            }
        )


def _enrich_summary(summary: dict[str, Any], *, full_flags: bool = False) -> dict[str, Any]:
    node_id = summary["id"]
    active = _contains_active.get(node_id, [])
    summary["contains"] = active
    if full_flags and node_id in _contains_flags:
        summary["contains_flags"] = _contains_flags[node_id]
    return summary


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _index
    if CARBONARA_AVAILABLE:
        load_fdc_map_cache()
    _init_contains()
    _index = FoodOnIndex.from_owl(cache_path=CACHE_PATH)
    yield
    _index = None
    _contains_active.clear()
    _contains_flags.clear()
    _contains_slugs.clear()
    _contains_summary.clear()


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
    if not CARBONARA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Carbonara explorer requires database setup")
    carbonara_html = STATIC_DIR / "carbonara.html"
    if not carbonara_html.is_file():
        raise HTTPException(status_code=404, detail="Carbonara page not found")
    return FileResponse(carbonara_html)


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
    if not CARBONARA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Composition API requires database setup")

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
        "contains_loaded": bool(_contains_slugs),
        "contains_node_count": len(_contains_flags),
        "contains_tagged_count": len(_contains_active),
        "carbonara_available": CARBONARA_AVAILABLE,
    }


@app.get("/api/contains/summary")
def contains_summary():
    if not _contains_slugs:
        raise HTTPException(status_code=404, detail="Contains table not loaded")
    return {
        "contains_slugs": _contains_slugs,
        "tagged_counts": _contains_summary.get("tagged_counts", {}),
        "nodes": _contains_summary.get("nodes", len(_contains_flags)),
    }


@app.get("/api/contains/{slug}/browse-roots")
def contains_browse_roots(slug: str):
    """FoodOn ancestor roots for a contains dimension — use as subtree entry points."""
    if slug not in _contains_slugs:
        raise HTTPException(status_code=404, detail=f"Unknown contains slug: {slug}")

    index = _require_index()
    roots_meta = [
        row
        for row in _contains_summary.get("ancestor_roots", [])
        if row.get("contains_slug") == slug and row.get("status") == "ok"
    ]

    if not roots_meta:
        from diet_tags_core import load_diet_tags

        registry = load_diet_tags()
        trigger = registry.contains.get(slug)
        if trigger is None:
            raise HTTPException(status_code=404, detail=f"No browse roots for: {slug}")
        roots_meta = [
            {"ancestor_id": anc_id, "contains_slug": slug, "status": "ok"}
            for anc_id in trigger.foodon_ancestors
        ]

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in roots_meta:
        ancestor_id = str(row["ancestor_id"])
        if ancestor_id in seen:
            continue
        seen.add(ancestor_id)
        summary = index.node_summary(ancestor_id)
        if summary is None:
            continue
        enriched = _enrich_summary(summary)
        enriched["tagged_descendant_count"] = int(row.get("descendant_count", 0))
        nodes.append(enriched)

    nodes.sort(key=lambda n: n.get("label", "").lower())
    return {"slug": slug, "nodes": nodes}


@app.get("/api/roots")
def roots():
    index = _require_index()
    return {
        "nodes": [
            _enrich_summary(index.node_summary(node_id))
            for node_id in index.preferred_roots()
        ],
    }


@app.get("/api/nodes/{node_id}")
def node_detail(node_id: str):
    index = _require_index()
    summary = index.node_summary(node_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    return _enrich_summary(summary, full_flags=True)


@app.get("/api/nodes/{node_id}/children")
def node_children(node_id: str):
    index = _require_index()
    if node_id not in index.labels:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    return {
        "id": node_id,
        "label": index.labels[node_id],
        "children": [
            _enrich_summary(child)
            for child in index.child_summaries(node_id)
        ],
    }


@app.get("/api/nodes/{node_id}/ancestors")
def node_ancestors(node_id: str):
    index = _require_index()
    if node_id not in index.labels:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    path = index.ancestry_path(node_id)
    return {
        "id": node_id,
        "ancestors": [
            _enrich_summary(index.node_summary(ancestor_id))
            for ancestor_id in path
        ],
    }


@app.get("/api/search")
def search(
    q: str = Query(min_length=1),
    limit: int = Query(default=25, ge=1, le=100),
):
    index = _require_index()
    results = [_enrich_summary(hit) for hit in index.search(q, limit=limit)]
    return {"query": q, "results": results}


def main():
    import uvicorn

    uvicorn.run("foodon_web.server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
