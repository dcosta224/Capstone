"""Global spacing between OpenAI calls to stay under 500 RPM."""

from __future__ import annotations

import asyncio
import threading
import time
import weakref

LLM_CALL_INTERVAL_SEC = 60 / 500

_loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)
_async_last = 0.0
_async_last_lock = threading.Lock()
_sync_lock = threading.Lock()
_sync_last = 0.0


def _async_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _loop_locks[loop] = lock
    return lock


async def throttle_llm_async() -> None:
    global _async_last
    async with _async_lock():
        with _async_last_lock:
            now = time.monotonic()
            wait = LLM_CALL_INTERVAL_SEC - (now - _async_last)
            if wait > 0:
                await asyncio.sleep(wait)
            _async_last = time.monotonic()


def throttle_llm_sync() -> None:
    global _sync_last
    with _sync_lock:
        now = time.monotonic()
        wait = LLM_CALL_INTERVAL_SEC - (now - _sync_last)
        if wait > 0:
            time.sleep(wait)
        _sync_last = time.monotonic()
