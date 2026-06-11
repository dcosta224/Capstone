"""FastAPI server for MVP recipe recommendation pipeline."""

from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from db import load_dotenv
from mvp_corpus_cache import corpus_status, warm_mvp_corpus
from mvp_nutrient_fit import clamp_fraction_bounds
from mvp_pipeline import PipelineEvent, UserQuery, get_embedding_model, run_pipeline

load_dotenv()

app = FastAPI(title="Recipe MVP", version="0.1.0")


@app.on_event("startup")
def startup_warm_cache() -> None:
    """Preload corpus + embedding model so demo requests stay fast."""
    warm_mvp_corpus()
    get_embedding_model()
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class RecommendRequest(BaseModel):
    taste_text: str = Field(min_length=1)
    kcal_min: float = Field(ge=0)
    kcal_max: float = Field(ge=0)
    fat_frac_min: float = Field(ge=0, le=1)
    fat_frac_max: float = Field(ge=0, le=1)
    carb_frac_min: float = Field(ge=0, le=1)
    carb_frac_max: float = Field(ge=0, le=1)
    protein_frac_min: float = Field(ge=0, le=1)
    protein_frac_max: float = Field(ge=0, le=1)
    w_semantic: float = Field(default=0.5, ge=0, le=1)
    w_nutrient: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=10, ge=1, le=20)


def _build_query(req: RecommendRequest) -> UserQuery:
    f0, f1, c0, c1, p0, p1 = clamp_fraction_bounds(
        req.fat_frac_min,
        req.fat_frac_max,
        req.carb_frac_min,
        req.carb_frac_max,
        req.protein_frac_min,
        req.protein_frac_max,
    )
    return UserQuery(
        taste_text=req.taste_text,
        kcal_min=req.kcal_min,
        kcal_max=req.kcal_max,
        fat_frac_min=f0,
        fat_frac_max=f1,
        carb_frac_min=c0,
        carb_frac_max=c1,
        protein_frac_min=p0,
        protein_frac_max=p1,
        w_semantic=req.w_semantic,
        w_nutrient=req.w_nutrient,
        top_k=req.top_k,
    )


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "cache": corpus_status()}


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    query = _build_query(req)

    def event_stream():
        q: queue.Queue = queue.Queue()

        def on_event(ev: PipelineEvent) -> None:
            if ev.stage == "done":
                return
            q.put(
                _sse_event(
                    "stage",
                    {"stage": ev.stage, "seq": ev.seq, "payload": ev.payload},
                )
            )

        def run() -> None:
            try:
                result = run_pipeline(query, on_event=on_event, log_to_db=True)
                q.put(_sse_event("done", {"stage": "done", "payload": result}))
            except Exception as exc:
                q.put(_sse_event("error", {"error": str(exc)}))
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
