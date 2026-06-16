"""Live progress + streaming artifacts for feasibility / Colab OSS runs."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from colab_s3 import parse_s3_uri
from judge_checkpoint import (
    append_judge_jsonl,
    compact_jsonl_to_parquet,
    load_judge_checkpoint,
)
from progress_utils import force_std_tqdm

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
    """Write progress.json, judge_stream.jsonl, periodic parquet compacts, optional quiet tqdm."""

    def __init__(
        self,
        out_dir: Path,
        *,
        run_id: str | None = None,
        quiet: bool = False,
        flush_every: int = 10,
        parquet_compact_every: int = 10,
        s3_sync_every: int = 10,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.out_dir.name
        self.quiet = quiet
        self.flush_every = max(1, flush_every)
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
        self._prompt_done = 0
        self._prompt_total: int | None = None
        self._rows_since_compact = 0
        self._rows_since_s3 = 0
        self._t0 = time.perf_counter()
        self._stats: dict[str, Any] = {
            "prompts_done": 0,
            "judge_errors": 0,
            "judge_fdc_matched": 0,
            "judge_abstained": 0,
            "judge_agreed": 0,
            "enrichment_errors": 0,
            "portion_llm_picks": 0,
        }
        self._pbar: Any = None
        if self.quiet:
            warnings.filterwarnings("ignore")
            force_std_tqdm()
            from tqdm import tqdm

            self._pbar = tqdm(
                total=0,
                desc="feasibility",
                unit="prompt",
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.5,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
        self._write_progress()

    @contextmanager
    def quiet_context(self) -> Iterator[None]:
        """Suppress stdout during pipeline phases (tqdm uses stderr)."""
        if not self.quiet:
            yield
            return
        saved = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stdout = saved

    def add_prompt_budget(self, n: int) -> None:
        if n <= 0:
            return
        self._prompt_total = (self._prompt_total or 0) + n
        if self._pbar is not None:
            self._pbar.total = self._prompt_total
            self._pbar.refresh()

    def set_phase(self, name: str, *, total: int | None = None) -> None:
        now = _utc_now()
        self._phase = name
        self._phase_started_at = now
        if name == "judging":
            self._judging_total = total
            self._judging_done = 0
            self._judging_t0 = time.perf_counter()
        if not self.quiet:
            msg = f"=== phase: {name}"
            if total is not None:
                msg += f" ({total:,} items)"
            msg += " ==="
            print(msg, flush=True)
        self._append_phase_log({"event": "phase_start", "phase": name, "total": total})
        self._write_progress()
        self._refresh_bar_postfix()

    def record_chunk_progress(self, phase: str, done: int, total: int) -> None:
        self._phase = phase
        if not self.quiet:
            pass  # no per-chunk console output in quiet mode
        self._write_progress()

    def record_enrichment_row(
        self,
        *,
        done: int,
        total: int,
        ingredient_norm: str,
        error: str | None = None,
    ) -> None:
        if error:
            self._stats["enrichment_errors"] += 1
        if not self.quiet:
            ts = _iso(_utc_now())
            flag = "ERR" if error else "OK"
            print(
                f"[enrich {done}/{total}] {ts} {ingredient_norm[:48]!r} {flag}",
                flush=True,
            )
        self._tick_prompt("enrich")

    def record_portion_llm_pick(self) -> None:
        self._stats["portion_llm_picks"] += 1
        self.add_prompt_budget(1)
        self._tick_prompt("portion")

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
        if exp.get("llm_error") is not None:
            self._stats["judge_errors"] += 1
        if exp.get("llm_fdc_id") is not None:
            self._stats["judge_fdc_matched"] += 1
        if exp.get("llm_abstained"):
            self._stats["judge_abstained"] += 1
        if exp.get("llm_agrees_with_staged"):
            self._stats["judge_agreed"] += 1
        self._tick_prompt("judge", last_row=row)
        if self._rows_since_compact >= self.parquet_compact_every:
            self.compact_parquet()
            self._rows_since_compact = 0
        return self._judging_done

    def _tick_prompt(self, phase: str, *, last_row: dict[str, Any] | None = None) -> None:
        self._prompt_done += 1
        self._stats["prompts_done"] = self._prompt_done
        if self._pbar is not None:
            self._pbar.update(1)
            self._refresh_bar_postfix(phase=phase)
        if self._prompt_done % self.flush_every == 0:
            self._write_progress(last_row=last_row)
            self.maybe_sync_s3(force=True)
            self._rows_since_s3 = 0
        elif last_row is not None and not self.quiet:
            self._write_progress(last_row=last_row)

    def _refresh_bar_postfix(self, *, phase: str | None = None) -> None:
        if self._pbar is None:
            return
        phase_label = phase or self._phase
        err = self._stats["judge_errors"]
        matched = self._stats["judge_fdc_matched"]
        self._pbar.set_postfix_str(f"{phase_label} err={err} fdc={matched}", refresh=False)

    def compact_parquet(self) -> int:
        if not self._jsonl_path.is_file():
            return len(load_judge_checkpoint(self._parquet_path))
        n = compact_jsonl_to_parquet(self._jsonl_path, self._parquet_path)
        if not self.quiet:
            print(f"  .. compacted judge_stream.jsonl -> {JUDGE_RAW_PARQUET} ({n} rows)", flush=True)
        return n

    def finalize(self) -> None:
        if self._jsonl_path.is_file():
            self.compact_parquet()
        self._append_phase_log({"event": "finalize", "phase": self._phase})
        self._write_progress()
        self.maybe_sync_s3(force=True)
        if self._pbar is not None:
            self._pbar.close()

    def maybe_sync_s3(self, *, force: bool) -> None:
        raw_prefix = os.environ.get("COLAB_PROGRESS_S3_PREFIX", "").strip()
        if not raw_prefix:
            return
        if not force and self._rows_since_s3 < self.s3_sync_every:
            return
        bucket, base_key = parse_s3_uri(raw_prefix.rstrip("/"))
        for name in (PROGRESS_JSON, JUDGE_STREAM_JSONL, JUDGE_RAW_PARQUET):
            local = self.out_dir / name
            if not local.is_file():
                continue
            key = f"{base_key}/{name}" if base_key else name
            try:
                from colab_s3 import upload_file_to_s3

                upload_file_to_s3(local, bucket, key)
            except Exception:
                pass

    def _append_phase_log(self, event: dict[str, Any]) -> None:
        event = {**event, "at": _iso(_utc_now()), "run_id": self.run_id}
        with self._phase_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=_json_default) + "\n")

    def _write_progress(self, *, last_row: dict[str, Any] | None = None) -> None:
        now = _utc_now()
        elapsed = time.perf_counter() - self._t0
        done = self._judging_done
        total = self._judging_total
        rate = 0.0
        eta = None
        if self._judging_t0 is not None and done > 0:
            judge_elapsed = time.perf_counter() - self._judging_t0
            rate = done / max(judge_elapsed, 1e-9)
            if total is not None and rate > 0:
                eta = int((total - done) / rate)

        fdc_rate = None
        if done > 0:
            fdc_rate = round(self._stats["judge_fdc_matched"] / done, 4)

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "phase": self._phase,
            "phase_started_at": _iso(self._phase_started_at),
            "updated_at": _iso(now),
            "elapsed_sec": round(elapsed, 1),
            "prompts": {
                "done": self._prompt_done,
                "total": self._prompt_total,
            },
            "stats": {
                **self._stats,
                "judge_fdc_match_rate": fdc_rate,
                "judge_error_rate": round(self._stats["judge_errors"] / done, 4) if done else None,
            },
        }
        if self._phase == "judging" and self._judging_total is not None:
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
