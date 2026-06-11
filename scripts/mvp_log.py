"""Persist MVP pipeline stages to mvp_logs schema."""

from __future__ import annotations

import json
import uuid
from typing import Any

from db import connect


def apply_mvp_logs_schema(cur) -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "sql" / "13_create_mvp_logs_schema.sql"
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    sql = "\n".join(lines)
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)


def start_run(taste_text: str, params: dict[str, Any]) -> str:
    run_id = str(uuid.uuid4())
    conn = connect()
    try:
        with conn.cursor() as cur:
            apply_mvp_logs_schema(cur)
            cur.execute(
                """
                INSERT INTO mvp_logs.query_runs (run_id, taste_text, params_json, status)
                VALUES (%s, %s, %s::jsonb, 'running')
                """,
                (run_id, taste_text, json.dumps(params)),
            )
        conn.commit()
    finally:
        conn.close()
    return run_id


def log_stage(run_id: str, stage: str, seq: int, payload: dict[str, Any]) -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mvp_logs.stage_events (run_id, stage, seq, payload_json)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (run_id, stage, seq, json.dumps(payload, default=str)),
            )
        conn.commit()
    finally:
        conn.close()


def finish_run(
    run_id: str,
    *,
    status: str = "done",
    chosen_recipe_id: int | None = None,
    error_message: str | None = None,
) -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE mvp_logs.query_runs
                SET status = %s, chosen_recipe_id = %s, error_message = %s
                WHERE run_id = %s
                """,
                (status, chosen_recipe_id, error_message, run_id),
            )
        conn.commit()
    finally:
        conn.close()
