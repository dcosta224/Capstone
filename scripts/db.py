"""Postgres connection helpers (Supabase pooler from repo .env)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import quote_plus

import psycopg2


def load_dotenv(env_path: Path | None = None) -> None:
    path = env_path or Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def database_url() -> str:
    load_dotenv()
    user = os.environ["PG_POOL_USER"]
    password = quote_plus(os.environ["PG_PASSWORD"])
    host = os.environ["PG_POOL_HOST"]
    database = os.environ.get("PG_DATABASE", "postgres")
    sslmode = os.environ.get("PG_SSL_MODE", "require")
    if os.environ.get("PG_PSQL_USE_TRANSACTION_POOLER_PORT", "0") == "1":
        port = os.environ.get("PG_POOL_TRANSACTION_PORT", "6543")
    else:
        port = os.environ.get("PG_POOL_SESSION_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


def connect(*, max_retries: int = 5, retry_delay: float = 1.0):
    """Open a Postgres connection with retries for flaky pooler handshakes."""
    url = database_url()
    delay = retry_delay
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(
                url,
                client_encoding="UTF8",
                connect_timeout=30,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
        except psycopg2.OperationalError as exc:
            last_exc = exc
            if attempt + 1 >= max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    assert last_exc is not None
    raise last_exc
