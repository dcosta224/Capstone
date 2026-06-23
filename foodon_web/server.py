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

from foodon_index import FoodOnIndex  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "foodon_index.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_index: FoodOnIndex | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _index
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
