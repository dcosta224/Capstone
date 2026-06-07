"""Kadin hybrid recipe deduplication (from exploration.ipynb)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from recipe_embedding import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    build_semantic_text,
    canonical_json_field,
    clean_ingredients,
    normalize_text,
    parse_json_list,
)

# Defaults from exploration.ipynb
DEFAULT_K = 10
DEFAULT_SEMANTIC_PAIR_THRESHOLD = 0.95
DEFAULT_VALIDATED_SEMANTIC = 0.92
DEFAULT_VALIDATED_JACCARD = 0.50
DEFAULT_VALIDATED_HYBRID = 0.75
HYBRID_WEIGHTS = (0.50, 0.35, 0.15)  # semantic, ingredient jaccard, title


@dataclass(frozen=True)
class DedupConfig:
    k: int = DEFAULT_K
    semantic_pair_threshold: float = DEFAULT_SEMANTIC_PAIR_THRESHOLD
    validated_semantic: float = DEFAULT_VALIDATED_SEMANTIC
    validated_jaccard: float = DEFAULT_VALIDATED_JACCARD
    validated_hybrid: float = DEFAULT_VALIDATED_HYBRID
    encode_batch_size: int = 256


@dataclass
class DedupPlan:
    total_rows: int
    exact_delete_ids: list[int]
    hybrid_delete_ids: list[int]
    keeper_ids: list[int]
    n_duplicate_pairs: int
    n_validated_pairs: int
    n_clusters: int


def prepare_recipe_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all rows without dropping duplicates (uses recipe_nlg.id)."""
    required = {"id", "title", "ingredients", "directions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    work = df.loc[:, ["id", "title", "ingredients", "directions"]].copy()
    work["id"] = work["id"].astype("int64")
    work["title_clean"] = work["title"].map(normalize_text)
    work["directions_clean"] = work["directions"].map(normalize_text)
    work["ingredients_clean"] = work["ingredients"].map(clean_ingredients)
    work["semantic_text"] = work.apply(
        lambda row: build_semantic_text(row["title_clean"], row["ingredients_clean"]),
        axis=1,
    )
    work["ingredient_count"] = work["ingredients_clean"].map(len)
    work["directions_length"] = work["directions_clean"].map(len)
    work["ingredients_key"] = work["ingredients"].map(canonical_json_field)
    work["directions_key"] = work["directions"].map(canonical_json_field)
    return work.reset_index(drop=True)


def exact_duplicate_ids(df: pd.DataFrame) -> list[int]:
    """IDs to remove when an earlier row shares title+ingredients+directions."""
    work = df.loc[:, ["id", "title", "ingredients_key", "directions_key"]].copy()
    work["title_key"] = work["title"].astype(str)
    ranked = work.sort_values("id")
    dup_mask = ranked.duplicated(
        subset=["title_key", "ingredients_key", "directions_key"],
        keep="first",
    )
    return ranked.loc[dup_mask, "id"].astype(int).tolist()


def ingredient_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def title_similarity(a: str, b: str) -> float:
    return fuzz.token_set_ratio(str(a), str(b)) / 100.0


def encode_semantic_texts(texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL.split("/", 1)[-1])
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vectors = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def semantic_duplicate_pairs(
    ids: np.ndarray,
    embeddings: np.ndarray,
    *,
    k: int,
    threshold: float,
) -> pd.DataFrame:
    index = build_faiss_index(embeddings)
    vectors = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(vectors)
    scores, neighbors = index.search(vectors, k)

    pairs: list[tuple[int, int, float]] = []
    for pos_i, recipe_id in enumerate(ids):
        for rank in range(1, k):
            pos_j = int(neighbors[pos_i, rank])
            score = float(scores[pos_i, rank])
            if score >= threshold:
                pairs.append((int(recipe_id), int(ids[pos_j]), score))

    if not pairs:
        return pd.DataFrame(columns=["recipe_id", "match_id", "similarity"])

    pairs_df = pd.DataFrame(pairs, columns=["recipe_id", "match_id", "similarity"])
    pairs_df["recipe_min"] = pairs_df[["recipe_id", "match_id"]].min(axis=1)
    pairs_df["recipe_max"] = pairs_df[["recipe_id", "match_id"]].max(axis=1)
    unique = (
        pairs_df.drop_duplicates(subset=["recipe_min", "recipe_max"])
        .loc[:, ["recipe_min", "recipe_max", "similarity"]]
        .rename(columns={"recipe_min": "recipe_id", "recipe_max": "match_id"})
    )
    return unique.reset_index(drop=True)


def build_hybrid_pairs(
    duplicate_pairs: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = clean_df.set_index("id")
    w_sem, w_ing, w_title = HYBRID_WEIGHTS
    rows: list[dict] = []

    for row in tqdm(
        duplicate_pairs.itertuples(index=False),
        total=len(duplicate_pairs),
        desc="hybrid scores",
    ):
        i = int(row.recipe_id)
        j = int(row.match_id)
        semantic_score = float(row.similarity)
        ingredient_score = ingredient_jaccard(
            meta.at[i, "ingredients_clean"],
            meta.at[j, "ingredients_clean"],
        )
        title_score = title_similarity(
            meta.at[i, "title_clean"],
            meta.at[j, "title_clean"],
        )
        hybrid_score = (
            w_sem * semantic_score + w_ing * ingredient_score + w_title * title_score
        )
        rows.append(
            {
                "recipe_id": i,
                "match_id": j,
                "semantic_score": semantic_score,
                "ingredient_jaccard": ingredient_score,
                "title_similarity": title_score,
                "hybrid_score": hybrid_score,
            }
        )

    return pd.DataFrame(rows)


def filter_validated_pairs(hybrid_pairs: pd.DataFrame, cfg: DedupConfig) -> pd.DataFrame:
    return hybrid_pairs[
        (hybrid_pairs["semantic_score"] >= cfg.validated_semantic)
        & (hybrid_pairs["ingredient_jaccard"] >= cfg.validated_jaccard)
        & (hybrid_pairs["hybrid_score"] >= cfg.validated_hybrid)
    ].copy()


def choose_keeper_ids(clean_df: pd.DataFrame, validated_pairs: pd.DataFrame) -> set[int]:
    all_ids = set(clean_df["id"].astype(int))

    if validated_pairs.empty:
        return all_ids

    graph = nx.Graph()
    graph.add_nodes_from(all_ids)
    graph.add_edges_from(
        validated_pairs[["recipe_id", "match_id"]].itertuples(index=False, name=None)
    )

    multi_clusters = [
        component
        for component in nx.connected_components(graph)
        if len(component) > 1
    ]

    clustered_ids: set[int] = set()
    keepers: set[int] = set(all_ids)

    meta = clean_df.set_index("id")
    for members in multi_clusters:
        clustered_ids.update(members)
        cluster_df = pd.DataFrame({"id": list(members)}).merge(
            clean_df.loc[:, ["id", "ingredient_count", "directions_length"]],
            on="id",
        )
        keeper = (
            cluster_df.sort_values(
                ["ingredient_count", "directions_length", "id"],
                ascending=[False, False, True],
            )
            .iloc[0]["id"]
        )
        keepers -= members
        keepers.add(int(keeper))

    return keepers


def plan_hybrid_dedup(
    clean_df: pd.DataFrame,
    embeddings: np.ndarray,
    cfg: DedupConfig,
) -> tuple[set[int], pd.DataFrame, pd.DataFrame]:
    ids = clean_df["id"].to_numpy(dtype=np.int64)
    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) != recipes ({len(ids)})"
        )
    if embeddings.shape[1] != EMBEDDING_DIMS:
        raise ValueError(f"Expected {EMBEDDING_DIMS} dims, got {embeddings.shape[1]}")

    duplicate_pairs = semantic_duplicate_pairs(
        ids,
        embeddings,
        k=cfg.k,
        threshold=cfg.semantic_pair_threshold,
    )
    hybrid_pairs = build_hybrid_pairs(duplicate_pairs, clean_df)
    validated_pairs = filter_validated_pairs(hybrid_pairs, cfg)
    keepers = choose_keeper_ids(clean_df, validated_pairs)
    delete_ids = set(ids.astype(int)) - keepers
    return delete_ids, hybrid_pairs, validated_pairs


def parse_pgvector(raw: object) -> np.ndarray:
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=np.float32)
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        return np.asarray(json.loads(text), dtype=np.float32)
    raise ValueError(f"Unrecognized vector format: {text[:80]!r}")
