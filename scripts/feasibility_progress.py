"""Live progress + streaming artifacts for feasibility / Colab OSS runs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from judge_checkpoint import (
    append_judge_jsonl,
    compact_jsonl_to_parquet,
    load_judge_checkpoint,
)

PROGRESS_JSON = "progress.json"
JUDGE_STREAM_JSONL = "judge_stream.jsonl"
PHASE_LOG_JSONL = "phase_log.jsonl"
JUDGE_RAW_PARQUET = "judge_matches_raw.parquet"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return _iso(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class FeasibilityProgressWriter:
    """Write progress.json, judge_stream.jsonl, and periodic parquet compacts."""

    def __init__(
        self,
        out_dir: Path,
        *,
        run_id: str | None = None,
        parquet_compact_every: int = 50,
        s3_sync_every: int = 50,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.out_dir.name
        self.parquet_compact_every = max(1, parquet_compact_every)
        self.s3_sync_every = max(1, s3_sync_every)
        self._jsonl_path = self.out_dir / JUDGE_STREAM_JSONL
        self._parquet_path = self.out_dir / JUDGE_RAW_PARQUET
        self._progress_path = self.out_dir / PROGRESS_JSON
        self._phase_log_path = self.out_dir / PHASE_LOG_JSONL
        self._phase = "init"
        self._phase_started_at = _utc_now()
        self._judging_total: int | None = None
        self._judging_done = 0
        self._judging_t0: float | None = None
        self._rows_since_compact = 0
        self._rows_since_s3 = 0
        self._write_progress()

    def set_phase(self, name: str, *, total: int | None = None) -> None:
        now = _utc_now()
        self._phase = name
        self._phase_started_at = now
        if name == "judging":
            self._judging_total = total
            self._judging_done = 0
            self._judging_t0 = time.perf_counter()
        msg = f"=== phase: {name}"
        if total is not None:
            msg += f" ({total:,} items)"
        msg += " ==="
        print(msg, flush=True)
        self._append_phase_log({"event": "phase_start", "phase": name, "total": total})
        self._write_progress()

    def record_chunk_progress(self, phase: str, done: int, total: int) -> None:
        self._phase = phase
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "phase": phase,
            "updated_at": _iso(_utc_now()),
            phase: {"done": done, "total": total},
        }
        self._progress_path.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        self.maybe_sync_s3(force=False)

    def record_enrichment_row(
        self,
        *,
        done: int,
        total: int,
        ingredient_norm: str,
        error: str | None = None,
    ) -> None:
        ts = _iso(_utc_now())
        flag = "ERR" if error else "OK"
        print(
            f"[enrich {done}/{total}] {ts} {ingredient_norm[:48]!r} {flag}",
            flush=True,
        )
        self.record_chunk_progress("amount_classify", done, total)

    def record_judge_row(self, exp: dict[str, Any]) -> int:
        row = dict(exp)
        if "inferred_at" not in row:
            ts = row.get("ts")
            if isinstance(ts, datetime):
                row["inferred_at"] = _iso(ts)
            else:
                row["inferred_at"] = _iso(_utc_now())
        append_judge_jsonl(self._jsonl_path, row)
        self._judging_done += 1
        self._rows_since_compact += 1
        self._rows_since_s3 += 1
        if self._rows_since_compact >= self.parquet_compact_every:
            self.compact_parquet()
            self._rows_since_compact = 0
        self._write_progress(last_row=row)
        if self._rows_since_s3 >= self.s3_sync_every:
            self.maybe_sync_s3(force=False)
            self._rows_since_s3 = 0
        return self._judging_done

    def compact_parquet(self) -> int:
        if not self._jsonl_path.is_file():
            return len(load_judge_checkpoint(self._parquet_path))
        n = compact_jsonl_to_parquet(self._jsonl_path, self._parquet_path)
        print(f"  .. compacted judge_stream.jsonl -> {JUDGE_RAW_PARQUET} ({n} rows)", flush=True)
        return n

    def finalize(self) -> None:
        if self._jsonl_path.is_file():
            self.compact_parquet()
        self._append_phase_log({"event": "finalize", "phase": self._phase})
        self.maybe_sync_s3(force=True)

    def maybe_sync_s3(self, *, force: bool) -> None:
        prefix = os.environ.get("COLAB_PROGRESS_S3_PREFIX", "").strip()
        if not prefix:
            return
        if not force and self._rows_since_s3 < self.s3_sync_every:
            return
        prefix = prefix.rstrip("/") + "/"
        for name in (PROGRESS_JSON, JUDGE_STREAM_JSONL, JUDGE_RAW_PARQUET):
            local = self.out_dir / name
            if not local.is_file():
                continue
            dest = f"{prefix}{name}"
            try:
                subprocess.run(
                    ["aws", "s3", "cp", str(local), dest],
                    check=False,
                    capture_output=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError):
                pass

    def _append_phase_log(self, event: dict[str, Any]) -> None:
        event = {**event, "at": _iso(_utc_now()), "run_id": self.run_id}
        with self._phase_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=_json_default) + "\n")

    def _write_progress(self, *, last_row: dict[str, Any] | None = None) -> None:
        now = _utc_now()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "phase": self._phase,
            "phase_started_at": _iso(self._phase_started_at),
            "updated_at": _iso(now),
        }
        if self._phase == "judging" and self._judging_total is not None:
            done = self._judging_done
            total = self._judging_total
            rate = 0.0
            eta = None
            if self._judging_t0 is not None and done > 0:
                elapsed = time.perf_counter() - self._judging_t0
                rate = done / max(elapsed, 1e-9)
                eta = int((total - done) / rate) if rate > 0 else None
            payload["judging"] = {
                "done": done,
                "total": total,
                "rate_per_sec": round(rate, 3),
                "eta_sec": eta,
            }
        if last_row is not None:
            payload["last_row"] = {
                "recipe_id": last_row.get("recipe_id"),
                "ingredient_idx": last_row.get("ingredient_idx"),
                "inferred_at": last_row.get("inferred_at"),
                "llm_fdc_id": last_row.get("llm_fdc_id"),
                "grams_status": last_row.get("grams_status"),
            }
        self._progress_path.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
