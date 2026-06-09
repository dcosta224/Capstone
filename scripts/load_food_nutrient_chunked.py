#!/usr/bin/env python3
"""Load food_nutrient.csv into usda.food_nutrient in chunks (avoids statement timeouts)."""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import connect, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "Data" / "All_Food_Data_April_2026" / "food_nutrient.csv"

COLUMNS = [
    "id",
    "fdc_id",
    "nutrient_id",
    "amount",
    "data_points",
    "derivation_id",
    "min",
    "max",
    "median",
    "loq",
    "footnote",
    "min_year_acquired",
    "percent_daily_value",
]

COPY_SQL = (
    "COPY usda.food_nutrient ("
    + ", ".join(COLUMNS)
    + ") FROM STDIN WITH (FORMAT csv, NULL '', QUOTE '\"', ESCAPE '\"')"
)


def configure_session(cur) -> None:
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET lock_timeout = 0")
    cur.execute("SET search_path TO usda, public")


def load_chunked(csv_path: Path, chunk_size: int, truncate: bool) -> int:
    load_dotenv()
    total = 0
    with connect() as conn:
        with conn.cursor() as cur:
            configure_session(cur)
            if truncate:
                cur.execute("TRUNCATE usda.food_nutrient")
                conn.commit()
                print("Truncated usda.food_nutrient")

        for i, chunk in enumerate(
            pd.read_csv(
                csv_path,
                usecols=COLUMNS,
                dtype="string",
                chunksize=chunk_size,
            ),
            start=1,
        ):
            buf = StringIO()
            chunk.to_csv(buf, index=False, header=False)
            buf.seek(0)
            with connect() as conn:
                with conn.cursor() as cur:
                    configure_session(cur)
                    cur.copy_expert(COPY_SQL, buf)
                conn.commit()
            total += len(chunk)
            print(f"  chunk {i}: {total:,} rows", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Append instead of replacing existing rows",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    print(f"Loading {args.csv} (chunk_size={args.chunk_size:,})")
    n = load_chunked(args.csv, args.chunk_size, truncate=not args.no_truncate)
    print(f"Done: {n:,} rows loaded into usda.food_nutrient")


if __name__ == "__main__":
    main()
