"""Supabase (Postgres) persistence for the LLM ingredient-matching pilot.

One script execution = one experiment. Writes incremental checkpoints into the
`inference` schema:
- `inference.match_experiments_0`  one row per script execution (the experiment)
- `inference.match_inferences_0`   one row per individual LLM call (inference)
- `inference.match_candidates_0`   top-N retrieval near misses per inference
- `inference.spend_checks_0`       budget circuit-breaker audit trail

All inserts upsert on natural keys so a run can be resumed or re-flushed without
duplicating rows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2.extras

from db import connect

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "sql" / "30_create_inference_schema.sql"

INFERENCE_COLUMNS = [
    "run_id", "run_name", "model",
    "recipe_id", "ingredient_idx", "ingredient", "name", "preparation",
    "dequantified", "unit",
    "system_prompt", "prompt", "response",
    "llm_fdc_id", "llm_description", "llm_certainty", "llm_rationale",
    "llm_agrees_with_staged", "llm_abstained", "llm_error",
    "staged_fdc_id", "staged_description", "staged_match_score",
    "staged_match_quality", "staged_base_score", "staged_prep_score",
    "n_candidates_llm", "n_lexical_pool", "n_semantic_pool",
    "staged_top1_in_llm_candidates", "staged_top1_in_top10",
    "n_relevant_steps", "relevant_steps",
    "prompt_tokens", "completion_tokens", "total_tokens", "price_estimate_usd",
    "ts", "ts_year", "ts_month", "ts_date", "ts_time",
]

CANDIDATE_COLUMNS = [
    "run_id", "recipe_id", "ingredient_idx", "rank", "fdc_id", "data_type",
    "description", "lexical_dequant", "dequant_sem", "retrieval_score",
    "staged_final_score", "staged_base_score", "staged_prep_score",
    "in_llm_prompt", "is_staged_top1", "is_llm_pick", "ts",
]

EXPERIMENT_COLUMNS = [
    "run_id", "run_name", "model", "mlflow_run_id", "mlflow_experiment",
    "prompt_version", "seed", "n_recipes", "n_ingredients", "n_llm_calls",
    "n_llm_errors", "prompt_tokens_total", "completion_tokens_total",
    "total_tokens", "cost_input_usd", "cost_output_usd", "cost_total_usd",
    "abstain_rate", "error_rate", "agreement_rate",
    "staged_top1_in_llm_candidates_rate", "staged_top1_in_top10_rate",
    "certainty_mean", "certainty_median", "certainty_std",
    "certainty_p01", "certainty_p05", "certainty_p10", "certainty_p90",
    "elapsed_sec", "concurrency", "retrieval_config", "pricing",
    "sampled_recipe_ids", "status", "started_at", "finished_at",
]

_JSONB_EXPERIMENT_FIELDS = {"retrieval_config", "pricing", "sampled_recipe_ids"}


def split_timestamp(ts: datetime) -> dict[str, Any]:
    """Single timestamp plus the split parts stored alongside it."""
    return {
        "ts": ts,
        "ts_year": ts.year,
        "ts_month": ts.month,
        "ts_date": ts.day,
        "ts_time": ts.strftime("%H:%M:%S"),
    }


def apply_schema(conn) -> None:
    """Create the inference schema/tables if absent (idempotent)."""
    sql = SCHEMA_SQL.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _row_tuple(row: dict[str, Any], columns: list[str], jsonb_fields: set[str]) -> tuple:
    out = []
    for col in columns:
        val = row.get(col)
        if col in jsonb_fields and val is not None:
            val = psycopg2.extras.Json(val)
        out.append(val)
    return tuple(out)


def _upsert(
    conn,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    conflict_cols: list[str],
    *,
    jsonb_fields: set[str] | None = None,
    update: bool = True,
) -> int:
    if not rows:
        return 0
    jsonb_fields = jsonb_fields or set()
    cols_sql = ", ".join(columns)
    conflict_sql = ", ".join(conflict_cols)
    if update:
        updates = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in columns if c not in conflict_cols
        )
        action = f"DO UPDATE SET {updates}"
    else:
        action = "DO NOTHING"
    sql = (
        f"INSERT INTO {table} ({cols_sql}) VALUES %s "
        f"ON CONFLICT ({conflict_sql}) {action}"
    )
    values = [_row_tuple(r, columns, jsonb_fields) for r in rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values, page_size=max(len(values), 1))
    conn.commit()
    return len(values)


def upsert_inferences(conn, rows: list[dict[str, Any]]) -> int:
    """Upsert per-LLM-call rows into inference.match_inferences_0."""
    return _upsert(
        conn,
        "inference.match_inferences_0",
        INFERENCE_COLUMNS,
        rows,
        ["run_id", "recipe_id", "ingredient_idx"],
    )


def upsert_candidates(conn, rows: list[dict[str, Any]]) -> int:
    return _upsert(
        conn,
        "inference.match_candidates_0",
        CANDIDATE_COLUMNS,
        rows,
        ["run_id", "recipe_id", "ingredient_idx", "fdc_id"],
    )


def upsert_experiment(conn, experiment: dict[str, Any]) -> int:
    """Upsert the per-execution experiment row into inference.match_experiments_0."""
    return _upsert(
        conn,
        "inference.match_experiments_0",
        EXPERIMENT_COLUMNS,
        [experiment],
        ["run_id"],
        jsonb_fields=_JSONB_EXPERIMENT_FIELDS,
    )


SPEND_CHECK_COLUMNS = [
    "run_id", "check_ts", "calls_completed", "window_start", "seconds_since_last",
    "spend_since_last_usd", "rate_usd_per_min", "past_day_spend_usd",
    "daily_limit_usd", "rate_limit_usd_per_min", "tripped", "reason",
]


def past_day_spend(conn) -> float:
    """Global spend (USD) recorded in the last rolling 24 hours, across all runs."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(price_estimate_usd), 0) "
            "FROM inference.match_inferences_0 "
            "WHERE ts >= now() - interval '1 day'"
        )
        return float(cur.fetchone()[0])
    # connection used read-only; no commit needed


def spend_since(conn, since_ts: datetime) -> float:
    """Global spend (USD) recorded strictly after `since_ts`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(price_estimate_usd), 0) "
            "FROM inference.match_inferences_0 WHERE ts > %s",
            (since_ts,),
        )
        return float(cur.fetchone()[0])


def last_amount_window(conn, amount: float = 10.0, lookback_days: int = 7) -> dict[str, Any]:
    """Time window over which the most recent `amount` USD was spent (global).

    Returns window_start/window_end timestamps, the spend covered, whether the
    full `amount` was reached, and the duration in minutes.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(ts), max(ts), COALESCE(SUM(price_estimate_usd), 0) "
            "FROM inference.match_inferences_0 "
            "WHERE price_estimate_usd IS NOT NULL AND ts >= now() - (%s || ' days')::interval",
            (lookback_days,),
        )
        min_ts, max_ts, total = cur.fetchone()
        total = float(total or 0.0)

        window_start = min_ts
        complete = False
        if total >= amount:
            cur.execute(
                """
                SELECT ts FROM (
                    SELECT ts,
                           SUM(price_estimate_usd) OVER (
                               ORDER BY ts DESC ROWS UNBOUNDED PRECEDING
                           ) AS cum
                    FROM inference.match_inferences_0
                    WHERE price_estimate_usd IS NOT NULL
                      AND ts >= now() - (%s || ' days')::interval
                ) q
                WHERE cum >= %s
                ORDER BY ts DESC
                LIMIT 1
                """,
                (lookback_days, amount),
            )
            row = cur.fetchone()
            if row:
                window_start = row[0]
                complete = True

    duration_min = None
    if window_start is not None and max_ts is not None:
        duration_min = round((max_ts - window_start).total_seconds() / 60.0, 2)
    return {
        "window_start": window_start,
        "window_end": max_ts,
        "covered_usd": round(min(total, amount), 4) if complete else round(total, 4),
        "amount_requested_usd": amount,
        "reached_amount": complete,
        "duration_min": duration_min,
        "lookback_total_usd": round(total, 4),
    }


def insert_spend_check(conn, row: dict[str, Any]) -> int:
    return _upsert(
        conn,
        "inference.spend_checks_0",
        SPEND_CHECK_COLUMNS,
        [row],
        ["check_id"],
        update=False,
    )


def latest_spend_check(conn, run_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT check_ts, calls_completed FROM inference.spend_checks_0 "
            "WHERE run_id = %s ORDER BY check_ts DESC LIMIT 1",
            (run_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"check_ts": row[0], "calls_completed": row[1]}


def open_connection():
    """Open a Supabase connection and ensure the schema exists."""
    conn = connect()
    apply_schema(conn)
    return conn
