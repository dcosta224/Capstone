"""Semantic embedding index over FoodOn class labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from foodon_paths import EMBEDDING_MODEL, FOODON_EMBED_DIR, FOODON_INDEX_CACHE


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


_EMBED_MODEL = None


def _get_embed_model(model_name: str):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(model_name.split("/", 1)[-1])
    return _EMBED_MODEL


class FoodOnEmbedIndex:
    def __init__(
        self,
        node_ids: list[str],
        labels: list[str],
        embeddings: np.ndarray,
        *,
        model_name: str,
    ) -> None:
        self.node_ids = node_ids
        self.labels = labels
        self.embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        self.model_name = model_name

    @classmethod
    def build(
        cls,
        foodon_index: Any,
        *,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = 256,
    ) -> FoodOnEmbedIndex:
        from sentence_transformers import SentenceTransformer

        node_ids = sorted(foodon_index.labels.keys())
        labels = [foodon_index.labels[nid] for nid in node_ids]
        model = SentenceTransformer(model_name.split("/", 1)[-1])
        embs = model.encode(
            labels,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return cls(node_ids, labels, np.asarray(embs, dtype=np.float32), model_name=model_name)

    @classmethod
    def from_disk(cls, directory: Path | None = None) -> FoodOnEmbedIndex:
        directory = directory or FOODON_EMBED_DIR
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        node_ids = np.load(directory / "node_ids.npy", allow_pickle=True).tolist()
        labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        embs = np.load(directory / "embeddings.f32.npy")
        return cls(
            [str(x) for x in node_ids],
            [str(x) for x in labels],
            embs,
            model_name=str(manifest.get("model_name", EMBEDDING_MODEL)),
        )

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or FOODON_EMBED_DIR
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "node_ids.npy", np.array(self.node_ids, dtype=str))
        (directory / "labels.json").write_text(
            json.dumps(self.labels),
            encoding="utf-8",
        )
        np.save(directory / "embeddings.f32.npy", self.embeddings.astype(np.float32))
        manifest = {
            "model_name": self.model_name,
            "count": len(self.node_ids),
            "dims": int(self.embeddings.shape[1]),
            "foodon_index_cache": str(FOODON_INDEX_CACHE),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return directory

    def encode_query(self, text: str) -> np.ndarray:
        model = _get_embed_model(self.model_name)
        vec = model.encode([text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    def search(self, text: str, *, k: int = 20) -> list[dict[str, Any]]:
        q = self.encode_query(text)
        sims = self.embeddings @ q
        k = min(k, len(self.node_ids))
        if k <= 0:
            return []
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            {
                "id": self.node_ids[i],
                "label": self.labels[i],
                "score": float(sims[i]),
            }
            for i in top_idx
        ]
