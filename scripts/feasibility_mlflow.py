"""MLflow logging for portion_pipeline_feasibility runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ingredient_match_llm import MLFLOW_ARTIFACTS, MLFLOW_DB, MLFLOW_DIR

DEFAULT_EXPERIMENT = "portion_pipeline_feasibility"
VERSION_PARAM = "feasibility_version"


def _ensure_mlflow():
    import mlflow

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    return mlflow


def next_feasibility_version(experiment_name: str = DEFAULT_EXPERIMENT) -> int:
    """Return max existing feasibility_version in experiment + 1 (starts at 1)."""
    try:
        mlflow = _ensure_mlflow()
    except Exception:
        return 1

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return 1

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=10_000,
    )
    if runs.empty:
        return 1

    col = f"params.{VERSION_PARAM}"
    if col not in runs.columns:
        return 1

    versions = pd.to_numeric(runs[col], errors="coerce").dropna()
    if versions.empty:
        return 1
    return int(versions.max()) + 1


def _numeric_report_metrics(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in report.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[key] = float(val)
    return out


def log_feasibility_run(
    *,
    experiment_name: str,
    run_name: str,
    feasibility_version: int,
    report: dict[str, Any],
    params: dict[str, Any],
    artifact_paths: dict[str, Path],
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log one feasibility pipeline run; returns MLflow run id."""
    try:
        mlflow = _ensure_mlflow()
    except Exception as exc:
        print(f"MLflow unavailable, skipping logging: {exc}", flush=True)
        return None

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        tag_map = {
            "task": "portion_pipeline_feasibility",
            VERSION_PARAM: str(feasibility_version),
        }
        if tags:
            tag_map.update(tags)
        mlflow.set_tags(tag_map)

        log_params = {VERSION_PARAM: feasibility_version, **params}
        # MLflow params must be str; coerce non-str scalars.
        mlflow.log_params(
            {k: str(v) for k, v in log_params.items() if v is not None}
        )
        mlflow.log_metrics(_numeric_report_metrics(report))

        for path in artifact_paths.values():
            if path.is_file():
                mlflow.log_artifact(str(path))

        return run.info.run_id
