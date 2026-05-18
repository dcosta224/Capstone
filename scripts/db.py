"""Postgres connection helpers (Supabase pooler from repo .env)."""

from __future__ import annotations

import os
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


def connect():
    return psycopg2.connect(database_url())
