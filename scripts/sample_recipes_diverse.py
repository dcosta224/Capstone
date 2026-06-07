#!/usr/bin/env python3
"""Select a diversity-representative recipe sample from exported embeddings.

Pipeline:
  1. Overcluster with MiniBatchKMeans (default k=20,000)
  2. Pick one medoid per cluster (real recipe closest to centroid)
  3. Select final N recipes with rare-cluster reserve, sqrt-size proportional
     allocation, and MMR (or facility-location) diversity fill

Prerequisites:
  uv run python scripts/export_recipe_embeddings.py

Usage:
  uv run python scripts/sample_recipes_diverse.py
  uv run python scripts/sample_recipes_diverse.py --n-clusters 500 --n-sample 100 \\
      --embed-dir Data/recipes/embeddings   # smoke test on partial load
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_DIR = ROOT / "Data" / "recipes" / "embeddings"
DEFAULT_CLUSTER_DIR = ROOT / "Data" / "recipes" / "clusters"
DEFAULT_OUTPUT = ROOT / "Data" / "recipes" / "mvp_sample_1000.json"


def load_embedding_artifacts(embed_dir: Path) -> tuple[np.ndarray, np.memmap]:
    manifest = json.loads((embed_dir / "manifest.json").read_text())
    dims = int(manifest["dims"])
    count = int(manifest["count"])
    recipe_ids = np.load(embed_dir / "recipe_ids.npy")
    embeddings = np.memmap(
        embed_dir / manifest["embeddings_path"],
        dtype=np.float32,
        mode="r",
        shape=(count, dims),
    )
    if len(recipe_ids) != count:
        raise ValueError(f"recipe_ids length {len(recipe_ids)} != manifest count {count}")
    return recipe_ids, embeddings


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def run_kmeans(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    batch_size: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels int32 shape (n,), centroids float32 (k, d))."""
    from sklearn.cluster import MiniBatchKMeans

    n = len(embeddings)
    k = min(n_clusters, n)
    if k < n_clusters:
        print(
            f"Warning: capping n_clusters {n_clusters} → {k} (only {n:,} embeddings loaded)",
            flush=True,
        )

    print(f"MiniBatchKMeans: k={k:,}, n={n:,}, batch_size={batch_size}", flush=True)
    t0 = time.perf_counter()
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(batch_size, n),
        max_iter=max_iter,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )
    labels = km.fit_predict(np.asarray(embeddings, dtype=np.float32))
    centroids = km.cluster_centers_.astype(np.float32)
    centroids = _normalize_rows(centroids)
    elapsed = time.perf_counter() - t0
    print(f"Clustering done in {elapsed:.1f}s", flush=True)
    return labels.astype(np.int32), centroids


def compute_medoids(
    recipe_ids: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    medoid_max_members: int,
    seed: int,
) -> pd.DataFrame:
    """One medoid row per non-empty cluster."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_clusters = centroids.shape[0]

    for cid in range(n_clusters):
        member_idx = np.flatnonzero(labels == cid)
        if member_idx.size == 0:
            continue

        if member_idx.size > medoid_max_members:
            member_idx = rng.choice(member_idx, size=medoid_max_members, replace=False)

        cluster_emb = np.asarray(embeddings[member_idx], dtype=np.float32)
        centroid = centroids[cid]
        sims = cluster_emb @ centroid
        best_local = int(np.argmax(sims))
        best_idx = int(member_idx[best_local])

        rows.append(
            {
                "recipe_id": int(recipe_ids[best_idx]),
                "cluster_id": int(cid),
                "cluster_size": int(np.sum(labels == cid)),
                "medoid_score": round(float(sims[best_local]), 6),
                "embed_idx": best_idx,
            }
        )

    df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    print(f"Medoids: {len(df):,} (from {n_clusters:,} clusters)", flush=True)
    return df


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d) → (n, m)"""
    return a @ b.T


def _mmr_select(
    candidate_emb: np.ndarray,
    n_pick: int,
    lam: float,
    rng: np.random.Generator,
) -> list[int]:
    """Greedy MMR on row indices into candidate_emb."""
    n = len(candidate_emb)
    if n_pick <= 0 or n == 0:
        return []
    n_pick = min(n_pick, n)

    mean = candidate_emb.mean(axis=0)
    mean /= max(np.linalg.norm(mean), 1e-12)
    first = int(np.argmin(candidate_emb @ mean))
    selected = [first]
    selected_emb = candidate_emb[first : first + 1]

    while len(selected) < n_pick:
        sims = _cosine_sim(candidate_emb, selected_emb)  # (n, |sel|)
        max_sim = sims.max(axis=1)
        min_dist = 1.0 - max_sim
        mmr = lam * min_dist - (1.0 - lam) * max_sim
        mmr[selected] = -np.inf
        nxt = int(np.argmax(mmr))
        if nxt in selected:
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        selected.append(nxt)
        selected_emb = candidate_emb[selected]

    return selected


def _facility_location_select(
    candidate_emb: np.ndarray,
    n_pick: int,
    rng: np.random.Generator,
) -> list[int]:
    """Greedy max-min distance (cosine) selection."""
    n = len(candidate_emb)
    if n_pick <= 0 or n == 0:
        return []
    n_pick = min(n_pick, n)

    mean = candidate_emb.mean(axis=0)
    mean /= max(np.linalg.norm(mean), 1e-12)
    first = int(np.argmin(candidate_emb @ mean))
    selected = [first]
    selected_emb = candidate_emb[first : first + 1]

    while len(selected) < n_pick:
        sims = _cosine_sim(candidate_emb, selected_emb)
        max_sim = sims.max(axis=1)
        max_sim[selected] = 2.0
        nxt = int(np.argmin(max_sim))
        if nxt in selected:
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        selected.append(nxt)
        selected_emb = candidate_emb[selected]

    return selected


def select_final_sample(
    medoids: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_sample: int,
    rare_quota: int,
    proportional_quota: int,
    min_cluster_size_proportional: int,
    rare_percentile: float,
    mmr_lambda: float,
    selector: str,
    seed: int,
) -> pd.DataFrame:
    """Three-tier selection from medoid table."""
    rng = np.random.default_rng(seed)
    mmr_quota = max(0, n_sample - rare_quota - proportional_quota)
    if rare_quota + proportional_quota + mmr_quota != n_sample:
        mmr_quota = max(0, n_sample - rare_quota - proportional_quota)

    selected_embed_idx: set[int] = set()
    picks: list[dict] = []

    def _append(rows: pd.DataFrame, tier: str) -> None:
        for row in rows.itertuples(index=False):
            if row.embed_idx in selected_embed_idx:
                continue
            selected_embed_idx.add(int(row.embed_idx))
            picks.append(
                {
                    "recipe_id": int(row.recipe_id),
                    "cluster_id": int(row.cluster_id),
                    "cluster_size": int(row.cluster_size),
                    "medoid_score": float(row.medoid_score),
                    "embed_idx": int(row.embed_idx),
                    "tier": tier,
                }
            )

    # Tier 1: rare-cluster reserve (bottom percentile by cluster size)
    threshold = float(np.quantile(medoids["cluster_size"].values, rare_percentile))
    rare_pool = medoids[medoids["cluster_size"] <= threshold].copy()
    if len(rare_pool) > 0:
        rare_n = min(rare_quota, len(rare_pool))
        rare_pick = rare_pool.sample(n=rare_n, random_state=seed)
        _append(rare_pick, "rare")

    # Tier 2: sqrt-size proportional (exclude tiny clusters)
    prop_pool = medoids[
        (~medoids["embed_idx"].isin(selected_embed_idx))
        & (medoids["cluster_size"] >= min_cluster_size_proportional)
    ].copy()
    prop_n = min(proportional_quota, max(0, n_sample - len(picks)), len(prop_pool))
    if prop_n > 0 and len(prop_pool) > 0:
        weights = np.sqrt(prop_pool["cluster_size"].values.astype(np.float64))
        weights /= weights.sum()
        prop_local = rng.choice(len(prop_pool), size=prop_n, replace=False, p=weights)
        _append(prop_pool.iloc[prop_local], "proportional")

    # Tier 3: diversity fill
    remain = medoids[~medoids["embed_idx"].isin(selected_embed_idx)].copy()
    fill_n = min(n_sample - len(picks), len(remain))
    if fill_n > 0:
        cand_emb = _normalize_rows(
            np.asarray(embeddings[remain["embed_idx"].values], dtype=np.float32)
        )
        if selector == "facility_location":
            local_sel = _facility_location_select(cand_emb, fill_n, rng)
        else:
            local_sel = _mmr_select(cand_emb, fill_n, mmr_lambda, rng)
        _append(remain.iloc[local_sel], "mmr" if selector == "mmr" else "facility_location")

    out = pd.DataFrame(picks)
    if len(out) < n_sample and len(out) < len(medoids):
        # Top up from any remaining medoids if quotas couldn't fill (small corpus)
        extra = medoids[~medoids["embed_idx"].isin(selected_embed_idx)]
        need = min(n_sample - len(out), len(extra))
        if need > 0:
            _append(extra.sample(n=need, random_state=seed + 1), "topup")

    out = pd.DataFrame(picks).head(n_sample)
    tier_counts = out["tier"].value_counts().to_dict()
    print(f"Selected {len(out):,} recipes: {tier_counts}", flush=True)
    return out


def pairwise_cosine_stats(embeddings: np.ndarray, embed_idx: np.ndarray) -> dict[str, float]:
    """Mean and min pairwise cosine distance on selected subset (sampled if large)."""
    sel = _normalize_rows(np.asarray(embeddings[embed_idx], dtype=np.float32))
    n = len(sel)
    if n < 2:
        return {"pairwise_cosine_dist_mean": 0.0, "pairwise_cosine_dist_min": 0.0}

    max_pairs = 200_000
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(500, n), replace=False)
        sel = sel[idx]

    sims = sel @ sel.T
    np.fill_diagonal(sims, 0.0)
    dists = 1.0 - sims
    triu = dists[np.triu_indices(len(sel), k=1)]
    return {
        "pairwise_cosine_dist_mean": round(float(triu.mean()), 4),
        "pairwise_cosine_dist_min": round(float(triu.min()), 4),
    }


def fetch_ingredient_counts(recipe_ids: list[int]) -> pd.Series:
    load_dotenv()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.recipe_id, f.ingredient_count
                FROM recipe.recipe_nlg_features f
                WHERE f.recipe_id = ANY(%s)
                """,
                (recipe_ids,),
            )
            rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({int(r[0]): int(r[1]) for r in rows})


def build_report(
    selected: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    n_embeddings: int,
    seed: int,
    selector: str,
    mmr_lambda: float,
) -> dict:
    tier_counts = selected["tier"].value_counts().to_dict()
    stats = pairwise_cosine_stats(embeddings, selected["embed_idx"].values)

    ing_counts = fetch_ingredient_counts(selected["recipe_id"].tolist())
    ing_summary: dict[str, float | None] = {
        "ingredient_count_mean": None,
        "ingredient_count_median": None,
    }
    if len(ing_counts) > 0:
        ing_summary["ingredient_count_mean"] = round(float(ing_counts.mean()), 2)
        ing_summary["ingredient_count_median"] = round(float(ing_counts.median()), 2)

    cluster_size_bins = pd.cut(
        selected["cluster_size"],
        bins=[0, 3, 10, 50, 200, 10_000],
        labels=["1-3", "4-10", "11-50", "51-200", "200+"],
    )
    occupancy = cluster_size_bins.value_counts().sort_index().to_dict()
    occupancy = {str(k): int(v) for k, v in occupancy.items()}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": n_embeddings,
        "n_clusters": n_clusters,
        "n_selected": len(selected),
        "seed": seed,
        "selector": selector,
        "mmr_lambda": mmr_lambda if selector == "mmr" else None,
        "tier_counts": tier_counts,
        "cluster_size_occupancy": occupancy,
        **stats,
        **ing_summary,
    }


def save_cluster_cache(
    cluster_dir: Path,
    n_clusters: int,
    labels: np.ndarray,
    centroids: np.ndarray,
    medoids: pd.DataFrame,
) -> None:
    cluster_dir.mkdir(parents=True, exist_ok=True)
    prefix = cluster_dir / f"k{n_clusters}"
    np.save(f"{prefix}_labels.npy", labels)
    np.save(f"{prefix}_centroids.npy", centroids)
    medoids.to_parquet(f"{prefix}_medoids.parquet", index=False)
    print(f"Cached clusters → {cluster_dir}", flush=True)


def load_cluster_cache(
    cluster_dir: Path,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    prefix = cluster_dir / f"k{n_clusters}"
    labels_path = Path(f"{prefix}_labels.npy")
    centroids_path = Path(f"{prefix}_centroids.npy")
    medoids_path = Path(f"{prefix}_medoids.parquet")
    if labels_path.is_file() and centroids_path.is_file() and medoids_path.is_file():
        labels = np.load(labels_path)
        centroids = np.load(centroids_path)
        medoids = pd.read_parquet(medoids_path)
        print(f"Loaded cached clusters from {cluster_dir}", flush=True)
        return labels, centroids, medoids
    return None


def run_pipeline(
    *,
    embed_dir: Path,
    cluster_dir: Path,
    output: Path,
    n_clusters: int,
    n_sample: int,
    seed: int,
    rare_quota: int,
    proportional_quota: int,
    min_cluster_size_proportional: int,
    rare_percentile: float,
    mmr_lambda: float,
    selector: str,
    kmeans_batch_size: int,
    kmeans_max_iter: int,
    medoid_max_members: int,
    skip_cluster: bool,
) -> dict:
    recipe_ids, embeddings = load_embedding_artifacts(embed_dir)
    n = len(recipe_ids)
    manifest = json.loads((embed_dir / "manifest.json").read_text())

    cached = load_cluster_cache(cluster_dir, n_clusters) if skip_cluster else None
    if cached is not None:
        labels, centroids, medoids = cached
    else:
        labels, centroids = run_kmeans(
            embeddings,
            n_clusters=n_clusters,
            seed=seed,
            batch_size=kmeans_batch_size,
            max_iter=kmeans_max_iter,
        )
        medoids = compute_medoids(
            recipe_ids,
            embeddings,
            labels,
            centroids,
            medoid_max_members=medoid_max_members,
            seed=seed,
        )
        save_cluster_cache(cluster_dir, centroids.shape[0], labels, centroids, medoids)

    k_eff = centroids.shape[0]
    selected = select_final_sample(
        medoids,
        embeddings,
        n_sample=n_sample,
        rare_quota=rare_quota,
        proportional_quota=proportional_quota,
        min_cluster_size_proportional=min_cluster_size_proportional,
        rare_percentile=rare_percentile,
        mmr_lambda=mmr_lambda,
        selector=selector,
        seed=seed,
    )

    report = build_report(
        selected,
        embeddings,
        n_clusters=k_eff,
        n_embeddings=n,
        seed=seed,
        selector=selector,
        mmr_lambda=mmr_lambda,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output.with_suffix(".parquet")
    report_path = output.with_name(output.stem + "_report.json")

    manifest_out = {
        "recipe_ids": selected["recipe_id"].astype(int).tolist(),
        "n_sample": len(selected),
        "n_clusters": k_eff,
        "n_embeddings": n,
        "seed": seed,
        "selector": selector,
        "mmr_lambda": mmr_lambda,
        "rare_quota": rare_quota,
        "proportional_quota": proportional_quota,
        "embedding_model": manifest.get("model"),
        "generated_at": report["generated_at"],
        "tier_counts": report["tier_counts"],
    }
    output.write_text(json.dumps(manifest_out, indent=2) + "\n")
    selected.to_parquet(parquet_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote sample manifest → {output}", flush=True)
    print(f"Wrote audit parquet   → {parquet_path}", flush=True)
    print(f"Wrote report          → {report_path}", flush=True)
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diversity-based MVP recipe sample.")
    parser.add_argument("--embed-dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-clusters", type=int, default=20_000)
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rare-quota", type=int, default=150)
    parser.add_argument("--proportional-quota", type=int, default=700)
    parser.add_argument("--min-cluster-size-proportional", type=int, default=3)
    parser.add_argument("--rare-percentile", type=float, default=0.25,
                        help="Clusters at or below this size quantile are 'rare'.")
    parser.add_argument("--mmr-lambda", type=float, default=0.7,
                        help="MMR breadth weight (higher = more diversity).")
    parser.add_argument("--selector", choices=("mmr", "facility_location"), default="mmr")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--medoid-max-members", type=int, default=5000,
                        help="Subsample large clusters when scoring medoids.")
    parser.add_argument("--skip-cluster", action="store_true",
                        help="Reuse cached cluster labels/medoids if present.")
    args = parser.parse_args()

    if args.rare_quota + args.proportional_quota > args.n_sample:
        raise SystemExit("rare_quota + proportional_quota must be <= n_sample")

    run_pipeline(
        embed_dir=args.embed_dir,
        cluster_dir=args.cluster_dir,
        output=args.output,
        n_clusters=args.n_clusters,
        n_sample=args.n_sample,
        seed=args.seed,
        rare_quota=args.rare_quota,
        proportional_quota=args.proportional_quota,
        min_cluster_size_proportional=args.min_cluster_size_proportional,
        rare_percentile=args.rare_percentile,
        mmr_lambda=args.mmr_lambda,
        selector=args.selector,
        kmeans_batch_size=args.kmeans_batch_size,
        kmeans_max_iter=args.kmeans_max_iter,
        medoid_max_members=args.medoid_max_members,
        skip_cluster=args.skip_cluster,
    )


if __name__ == "__main__":
    main()
