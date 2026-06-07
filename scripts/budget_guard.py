"""Spend circuit breaker for the LLM ingredient-matching pilot.

Every `check_every` completed LLM calls (each call counted individually, so a
concurrency batch of 8 counts as 8), the guard queries Supabase for:

- global spend in the last rolling 24 hours, and
- the spend rate since the previous check (USD/minute).

If the daily spend exceeds `daily_limit_usd` OR the recent rate exceeds
`rate_limit_usd_per_min`, the breaker trips: the caller stops launching new
calls, drains in-flight calls, writes remaining results, and aborts.

Every check is logged to `inference.spend_checks_0` for a persistent audit trail.
All spend is derived from `inference.match_inferences_0.price_estimate_usd`, so
the accounting survives restarts and spans every run in the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import inference_store


@dataclass
class BudgetConfig:
    daily_limit_usd: float = 10.0
    rate_limit_usd_per_min: float = 0.50
    check_every: int = 100


class BudgetGuard:
    """Tallies completed calls and runs DB-backed spend checks on a cadence."""

    def __init__(self, conn, run_id: str, config: BudgetConfig):
        self.conn = conn
        self.run_id = run_id
        self.config = config
        # Rate window starts at construction (start of judging) unless a prior
        # check exists for this run_id (resume case).
        self.last_check_ts: datetime = datetime.now(timezone.utc)
        prior = inference_store.latest_spend_check(conn, run_id)
        if prior and prior.get("check_ts"):
            self.last_check_ts = prior["check_ts"]

    def should_check(self, completed: int) -> bool:
        return completed > 0 and completed % self.config.check_every == 0

    def check(self, completed: int) -> dict[str, Any]:
        """Run one spend check (blocking DB I/O); log it; return the verdict.

        Intended to be invoked via loop.run_in_executor on the DB worker thread.
        """
        cfg = self.config
        now = datetime.now(timezone.utc)
        past_day = inference_store.past_day_spend(self.conn)
        spend_recent = inference_store.spend_since(self.conn, self.last_check_ts)
        seconds = max((now - self.last_check_ts).total_seconds(), 1e-9)
        rate = spend_recent / (seconds / 60.0)

        reasons: list[str] = []
        if past_day > cfg.daily_limit_usd:
            reasons.append(
                f"past-day spend ${past_day:.2f} exceeds cap ${cfg.daily_limit_usd:.2f}"
            )
        if rate > cfg.rate_limit_usd_per_min:
            reasons.append(
                f"recent rate ${rate:.3f}/min exceeds cap "
                f"${cfg.rate_limit_usd_per_min:.2f}/min"
            )
        tripped = bool(reasons)
        reason = "; ".join(reasons) if reasons else None

        inference_store.insert_spend_check(self.conn, {
            "run_id": self.run_id,
            "check_ts": now,
            "calls_completed": completed,
            "window_start": self.last_check_ts,
            "seconds_since_last": round(seconds, 3),
            "spend_since_last_usd": round(spend_recent, 6),
            "rate_usd_per_min": round(rate, 6),
            "past_day_spend_usd": round(past_day, 6),
            "daily_limit_usd": cfg.daily_limit_usd,
            "rate_limit_usd_per_min": cfg.rate_limit_usd_per_min,
            "tripped": tripped,
            "reason": reason,
        })

        verdict = {
            "check_ts": now,
            "calls_completed": completed,
            "window_start": self.last_check_ts,
            "seconds_since_last": round(seconds, 3),
            "spend_since_last_usd": round(spend_recent, 6),
            "rate_usd_per_min": round(rate, 6),
            "past_day_spend_usd": round(past_day, 6),
            "tripped": tripped,
            "reason": reason,
        }
        self.last_check_ts = now
        return verdict

    def spend_window(self, amount: float) -> dict[str, Any]:
        """Window over which the most recent `amount` USD was spent (for summary)."""
        return inference_store.last_amount_window(self.conn, amount=amount)
