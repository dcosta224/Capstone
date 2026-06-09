#!/usr/bin/env python3
"""Run tiered ingredient parse experiment (lib_llm vs lib_rules_llm).

Logs two MLflow runs under one experiment and writes local error-analysis artifacts.

Usage:
    uv run python scripts/run_parse_experiment.py --n-recipes 100 --seed 42
    uv run python scripts/run_parse_experiment.py --no-llm --n-recipes 100 --seed 42
    uv run python scripts/run_parse_experiment.py --pipeline lib_llm --n-recipes 100
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from db import load_dotenv
from ingredient_parse_llm import DEFAULT_MODEL, PROMPT_VERSION
from ingredient_parse_pipeline import (
    RULES_CONFIDENCE_THRESHOLD,
    compute_pipeline_report,
    run_lib_llm,
    run_lib_rules_llm,
)
from parse_experiment_artifacts import (
    build_comparison,
    build_error_analysis,
    manifest_hash,
    write_comparison_artifacts,
    write_pipeline_artifacts,
)
from sample_recipes import load_sampled_recipes

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DIR = ROOT / "mlruns"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
DEFAULT_WORK_ROOT = ROOT / "scratch" / "parse_experiments"
MLFLOW_EXPERIMENT = "ingredient_quantity_parse"


def log_to_mlflow(
    *,
    experiment_name: str,
    run_name: str,
    report: dict[str, Any],
    artifact_paths: dict[str, Path],
    batch_id: str,
    eval_manifest: dict[str, Any],
) -> str | None:
    try:
        import mlflow
    except Exception as exc:
        print(f"MLflow unavailable, skipping: {exc}", flush=True)
        return None

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment_name)

    metric_keys = [
        "coverage_ok_rate",
        "coverage_measurable_rate",
        "unmeasurable_rate",
        "library_only_rate",
        "rules_resolved_rate",
        "llm_resolved_rate",
        "llm_call_rate",
        "still_unparsed_rate",
        "llm_error_rate",
        "llm_certainty_mean",
        "llm_certainty_median",
        "llm_certainty_p90",
        "n_ingredient_lines",
        "n_unique_ingredients",
        "n_llm_calls",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens",
        "cost_input_usd",
        "cost_output_usd",
        "cost_total_usd",
        "cost_avg_per_unique_ingredient_usd",
        "elapsed_sec",
    ]

    param_keys = [
        "pipeline_version",
        "model",
        "seed",
        "n_recipes",
        "n_ingredient_lines",
        "n_unique_ingredients",
        "prompt_version",
        "rules_confidence_threshold",
        "use_llm",
    ]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "task": "ingredient_quantity_parse",
            "pipeline_version": report.get("pipeline_version"),
            "prompt_version": PROMPT_VERSION,
            "eval_seed": str(report.get("seed")),
            "eval_batch_id": batch_id,
            "eval_manifest_hash": manifest_hash(eval_manifest),
        })
        params = {k: report.get(k) for k in param_keys if report.get(k) is not None}
        mlflow.log_params({k: str(v) for k, v in params.items()})
        metrics = {k: report[k] for k in metric_keys if report.get(k) is not None}
        mlflow.log_metrics(metrics)

        status_path = artifact_paths.get("parse_summary")
        if status_path and status_path.is_file():
            counts = {
                "final_parse_status_counts": report.get("final_parse_status_counts"),
                "final_method_counts": report.get("final_method_counts"),
                "library_parse_status_counts": report.get("library_parse_status_counts"),
            }
            counts_path = status_path.parent / "status_breakdown.json"
            counts_path.write_text(json.dumps(counts, indent=2) + "\n")
            artifact_paths = {**artifact_paths, "status_breakdown": counts_path}

        for path in artifact_paths.values():
            if path.is_file():
                mlflow.log_artifact(str(path))

        return run.info.run_id


def run_one_pipeline(
    pipeline: str,
    eval_df: pd.DataFrame,
    *,
    model: str,
    concurrency: int,
    use_llm: bool,
    n_recipes: int,
    seed: int,
    rules_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if pipeline == "lib_llm":
        return run_lib_llm(
            eval_df,
            model=model,
            concurrency=concurrency,
            use_llm=use_llm,
        )
    if pipeline == "lib_rules_llm":
        return run_lib_rules_llm(
            eval_df,
            model=model,
            concurrency=concurrency,
            use_llm=use_llm,
            rules_threshold=rules_threshold,
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-recipes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--pipeline",
        choices=["both", "lib_llm", "lib_rules_llm"],
        default="both",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM tier (baseline)")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--mlflow-experiment",
        default=MLFLOW_EXPERIMENT,
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=DEFAULT_WORK_ROOT,
    )
    parser.add_argument(
        "--rules-confidence-threshold",
        type=float,
        default=RULES_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help="Optional manifest with recipe_ids",
    )
    args = parser.parse_args()
    load_dotenv()

    t0 = time.perf_counter()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"{ts}_seed{args.seed}_n{args.n_recipes}"
    batch_dir = args.work_root / batch_id

    recipes, eval_df, sampled_ids = load_sampled_recipes(
        n=args.n_recipes,
        seed=args.seed,
        sample_manifest=args.sample_manifest,
    )
    eval_df = eval_df.copy()
    eval_df["ingredient_norm"] = eval_df["ingredient"].astype(str).str.strip().str.lower()

    eval_manifest = {
        "batch_id": batch_id,
        "seed": args.seed,
        "n_recipes": args.n_recipes,
        "recipe_ids": sampled_ids,
        "n_ingredient_lines": len(eval_df),
        "n_unique_ingredients": int(eval_df["ingredient_norm"].nunique()),
        "use_llm": not args.no_llm,
        "pipeline": args.pipeline,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "eval_manifest.json").write_text(
        json.dumps(eval_manifest, indent=2) + "\n"
    )

    pipelines = (
        ["lib_llm", "lib_rules_llm"]
        if args.pipeline == "both"
        else [args.pipeline]
    )

    results_by_pipeline: dict[str, pd.DataFrame] = {}
    reports_by_pipeline: dict[str, dict[str, Any]] = {}

    for pipeline in pipelines:
        print(f"\n=== Running pipeline: {pipeline} ===", flush=True)
        results_df, llm_calls_df, meta = run_one_pipeline(
            pipeline,
            eval_df,
            model=args.model,
            concurrency=args.concurrency,
            use_llm=not args.no_llm,
            n_recipes=len(sampled_ids),
            seed=args.seed,
            rules_threshold=args.rules_confidence_threshold,
        )
        report = compute_pipeline_report(
            results_df,
            llm_calls_df,
            model=args.model,
            meta=meta,
            n_recipes=len(sampled_ids),
            seed=args.seed,
            rules_threshold=args.rules_confidence_threshold,
        )

        other_df = results_by_pipeline.get("lib_llm") if pipeline == "lib_rules_llm" else None
        error_df = build_error_analysis(results_df, other_df=other_df)

        pipe_dir = batch_dir / pipeline
        artifact_paths = write_pipeline_artifacts(
            pipe_dir,
            results_df,
            llm_calls_df,
            report,
            error_df,
        )

        run_name = f"{pipeline}_{ts}"
        if not args.no_llm:
            run_name += "_llm"
        else:
            run_name += "_baseline"

        mlflow_run_id = None
        if not args.no_mlflow:
            mlflow_run_id = log_to_mlflow(
                experiment_name=args.mlflow_experiment,
                run_name=run_name,
                report=report,
                artifact_paths=artifact_paths,
                batch_id=batch_id,
                eval_manifest=eval_manifest,
            )
            if mlflow_run_id:
                print(f"MLflow run: {mlflow_run_id} ({run_name})", flush=True)

        print(json.dumps({k: report[k] for k in (
            "pipeline_version", "coverage_ok_rate", "library_only_rate",
            "rules_resolved_rate", "llm_resolved_rate", "llm_call_rate",
            "n_llm_calls", "cost_total_usd", "elapsed_sec",
        )}, indent=2), flush=True)

        results_by_pipeline[pipeline] = results_df
        reports_by_pipeline[pipeline] = report

    if "lib_llm" in results_by_pipeline and "lib_rules_llm" in results_by_pipeline:
        side, rules_rescued, disagreements, llm_only, comp_summary = build_comparison(
            results_by_pipeline["lib_llm"],
            results_by_pipeline["lib_rules_llm"],
            lib_llm_report=reports_by_pipeline["lib_llm"],
            lib_rules_report=reports_by_pipeline["lib_rules_llm"],
        )
        comp_paths = write_comparison_artifacts(
            batch_dir / "comparison",
            side,
            rules_rescued,
            disagreements,
            llm_only,
            comp_summary,
        )
        print("\n=== Comparison summary ===", flush=True)
        print(json.dumps(comp_summary, indent=2), flush=True)

        if not args.no_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
                mlflow.set_experiment(args.mlflow_experiment)
                with mlflow.start_run(run_name=f"comparison_{ts}") as run:
                    mlflow.set_tags({
                        "task": "ingredient_quantity_parse_comparison",
                        "eval_batch_id": batch_id,
                    })
                    mlflow.log_metrics({
                        k: float(v)
                        for k, v in comp_summary.items()
                        if isinstance(v, (int, float)) and v is not None
                    })
                    for path in comp_paths.values():
                        if path.is_file():
                            mlflow.log_artifact(str(path))
                    print(f"MLflow comparison run: {run.info.run_id}", flush=True)
            except Exception as exc:
                print(f"MLflow comparison logging skipped: {exc}", flush=True)

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\nDone in {elapsed}s — artifacts: {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
