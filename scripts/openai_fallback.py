"""OpenAI clients with sticky API-key failover on rate limits."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from db import load_dotenv
from llm_throttle import throttle_llm_async, throttle_llm_sync

RETRIES_PER_KEY = 3
BACKOFF_BASE_SEC = 0.5
BACKOFF_MAX_SEC = 8.0

_state_lock = threading.Lock()
_active_key_index = 0
_blocked_key_indices: set[int] = set()


def get_openai_api_keys() -> list[str]:
    load_dotenv()
    keys: list[str] = []
    for name in ("OPENAI_API_KEY", "OPENAI_API_KEY_2"):
        value = os.environ.get(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    if not keys:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY in .env"
        )
    return keys


def _is_daily_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "requests per day" in msg or "rpd" in msg


def _should_switch_key(exc: Exception) -> bool:
    msg = str(exc).lower()
    if _is_daily_limit(exc):
        return True
    if "rate_limit" in msg or type(exc).__name__ == "RateLimitError":
        return True
    if "insufficient_quota" in msg or "billing" in msg:
        return True
    status = getattr(exc, "status_code", None)
    return status in (401, 403, 429)


def _is_retryable_same_key(exc: Exception) -> bool:
    if _should_switch_key(exc):
        return not _is_daily_limit(exc)
    msg = str(exc).lower()
    if any(s in msg for s in ("timeout", "timed out", "connection", "server error", "503", "502")):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and int(status) >= 500


def _candidate_key_indices(n_keys: int) -> list[int]:
    with _state_lock:
        active = _active_key_index
        blocked = set(_blocked_key_indices)
    available = [i for i in range(n_keys) if i not in blocked]
    if not available:
        return list(range(n_keys))
    if active in available:
        return [active] + [i for i in available if i != active]
    return available


def _mark_key_exhausted(failed_idx: int, n_keys: int) -> None:
    global _active_key_index
    with _state_lock:
        _blocked_key_indices.add(failed_idx)
        remaining = [i for i in range(n_keys) if i not in _blocked_key_indices]
        if remaining:
            _active_key_index = remaining[0]


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(BACKOFF_BASE_SEC * (2**attempt), BACKOFF_MAX_SEC))


async def _backoff_sleep_async(attempt: int) -> None:
    await asyncio.sleep(min(BACKOFF_BASE_SEC * (2**attempt), BACKOFF_MAX_SEC))


def _execute_with_fallback(call_fn: Any, *, throttle_fn: Any) -> Any:
    from openai import OpenAI

    keys = get_openai_api_keys()
    last_exc: Exception | None = None
    for key_idx in _candidate_key_indices(len(keys)):
        client = OpenAI(api_key=keys[key_idx])
        for attempt in range(RETRIES_PER_KEY):
            try:
                throttle_fn()
                return call_fn(client)
            except Exception as exc:
                last_exc = exc
                if _should_switch_key(exc):
                    _mark_key_exhausted(key_idx, len(keys))
                    break
                if _is_retryable_same_key(exc) and attempt < RETRIES_PER_KEY - 1:
                    _backoff_sleep(attempt)
                    continue
                raise
    raise last_exc  # type: ignore[misc]


async def _execute_with_fallback_async(call_fn: Any) -> Any:
    from openai import AsyncOpenAI

    keys = get_openai_api_keys()
    last_exc: Exception | None = None
    for key_idx in _candidate_key_indices(len(keys)):
        client = AsyncOpenAI(api_key=keys[key_idx])
        for attempt in range(RETRIES_PER_KEY):
            try:
                await throttle_llm_async()
                return await call_fn(client)
            except Exception as exc:
                last_exc = exc
                if _should_switch_key(exc):
                    _mark_key_exhausted(key_idx, len(keys))
                    break
                if _is_retryable_same_key(exc) and attempt < RETRIES_PER_KEY - 1:
                    await _backoff_sleep_async(attempt)
                    continue
                raise
    raise last_exc  # type: ignore[misc]


class _FallbackAsyncCompletions:
    async def create(self, **kwargs: Any) -> Any:
        async def _call(client: Any) -> Any:
            return await client.chat.completions.create(**kwargs)

        return await _execute_with_fallback_async(_call)


class _FallbackAsyncChat:
    def __init__(self) -> None:
        self.completions = _FallbackAsyncCompletions()


class AsyncOpenAIFallback:
    def __init__(self) -> None:
        self.chat = _FallbackAsyncChat()


class _FallbackSyncCompletions:
    def create(self, **kwargs: Any) -> Any:
        def _call(client: Any) -> Any:
            return client.chat.completions.create(**kwargs)

        return _execute_with_fallback(_call, throttle_fn=throttle_llm_sync)


class _FallbackSyncChat:
    def __init__(self) -> None:
        self.completions = _FallbackSyncCompletions()


class SyncOpenAIFallback:
    def __init__(self) -> None:
        self.chat = _FallbackSyncChat()


def get_async_openai_client() -> AsyncOpenAIFallback:
    return AsyncOpenAIFallback()


def get_sync_openai_client() -> SyncOpenAIFallback:
    return SyncOpenAIFallback()
