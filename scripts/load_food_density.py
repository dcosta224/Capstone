#!/usr/bin/env python3
"""
Extract FAO/INFOODS food density table from food_density.pdf to CSV, then load
into the `conversions` schema on Supabase.

Usage:
  pip install -r requirements.txt
  python scripts/load_food_density.py
  python scripts/load_food_density.py --extract-only
  python scripts/load_food_density.py --load-only --csv Data/conversions/food_density.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import pdfplumber
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "Data" / "food_density.pdf"
DEFAULT_CSV = ROOT / "Data" / "conversions" / "food_density.csv"
SCHEMA_SQL = ROOT / "sql" / "20_create_conversions_schema.sql"

NUMERIC = re.compile(r"^[\d.]+(?:-[\d.]+)?$")
CSV_COLUMNS = [
    "food_group",
    "food_name",
    "density_g_per_ml",
    "specific_gravity",
    "biblio_id",
    "updated_version_2",
]


def compact_row(row: list) -> list[str]:
    return [
        str(cell).replace("\n", " ").strip()
        for cell in row
        if cell is not None and str(cell).strip()
    ]


def parse_data_row(parts: list[str]) -> tuple[str, str, str, str, str] | None:
    """Return (food_name, density, specific_gravity, biblio_id, updated)."""
    numeric_idx = next((i for i, p in enumerate(parts) if NUMERIC.match(p)), None)
    if numeric_idx is None:
        return None

    food_name = " ".join(parts[:numeric_idx]).strip()
    density = parts[numeric_idx]
    specific_gravity = ""
    biblio_id = ""
    updated = ""

    for token in parts[numeric_idx + 1 :]:
        if token == "x":
            updated = "x"
        elif NUMERIC.match(token) and not specific_gravity:
            specific_gravity = token
        elif not biblio_id:
            biblio_id = token
        elif NUMERIC.match(token):
            specific_gravity = token

    if not food_name:
        return None
    return food_name, density, specific_gravity, biblio_id, updated


def extract_pdf(pdf_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_group: str | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                header = " ".join(str(c or "") for c in table[0]).lower()
                if "food name" not in header:
                    continue

                for row in table[3:]:
                    parts = compact_row(row)
                    if not parts:
                        continue

                    if len(parts) == 1:
                        current_group = parts[0]
                        continue

                    if not any(NUMERIC.match(p) for p in parts):
                        if len(parts) == 1:
                            current_group = parts[0]
                        continue

                    parsed = parse_data_row(parts)
                    if parsed is None:
                        continue

                    food_name, density, sg, biblio, updated = parsed
                    records.append(
                        {
                            "food_group": current_group or "",
                            "food_name": food_name,
                            "density_g_per_ml": density,
                            "specific_gravity": sg,
                            "biblio_id": biblio,
                            "updated_version_2": "true" if updated == "x" else "false",
                        }
                    )

    return records


def write_csv(records: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def apply_schema_sql(cur, path: Path) -> None:
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            cur.execute(stmt)


def load_csv_to_db(csv_path: Path) -> int:
    with connect() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0")
            apply_schema_sql(cur, SCHEMA_SQL)

            copy_sql = (
                "COPY conversions.food_density "
                "(food_group, food_name, density_g_per_ml, specific_gravity, "
                "biblio_id, updated_version_2) "
                "FROM STDIN WITH (FORMAT csv, HEADER true, QUOTE '\"', ESCAPE '\"')"
            )
            with csv_path.open(encoding="utf-8") as f:
                cur.copy_expert(copy_sql, f)

            cur.execute("SELECT COUNT(*) FROM conversions.food_density")
            count = cur.fetchone()[0]
        conn.commit()
    return int(count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract food_density.pdf to CSV and load conversions schema"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only write CSV; do not load Postgres",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load existing CSV into Postgres",
    )
    args = parser.parse_args()
    load_dotenv()

    t0 = time.perf_counter()

    if not args.load_only:
        if not args.pdf.is_file():
            sys.exit(f"PDF not found: {args.pdf}")
        print(f"Extracting {args.pdf} …")
        records = extract_pdf(args.pdf)
        if not records:
            sys.exit("No food density rows extracted from PDF")
        write_csv(records, args.csv)
        print(f"Wrote {len(records):,} rows to {args.csv}")

    if not args.extract_only:
        if not args.csv.is_file():
            sys.exit(f"CSV not found: {args.csv} (run without --load-only first)")
        print(f"Loading {args.csv} into conversions.food_density …")
        count = load_csv_to_db(args.csv)
        print(f"Loaded {count:,} rows into conversions.food_density")

    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
