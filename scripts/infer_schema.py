#!/usr/bin/env python3
"""
Inspect the `usda` schema in Supabase/Postgres: tables, columns, PKs/FKs,
and inferred joins (explicit constraints + naming heuristics).

Usage:
  python scripts/infer_schema.py
  python scripts/infer_schema.py --schema usda --out scripts/usda_schema_inferred.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, load_dotenv

# FDC hub table and common extension pattern (fdc_id on satellite tables).
HUB_TABLE = "food"
HUB_PK = "fdc_id"

# Column name -> likely referenced table.pk (USDA FoodData Central conventions).
NAMED_JOINS: dict[str, tuple[str, str]] = {
    "fdc_id": (HUB_TABLE, HUB_PK),
    "food_id": (HUB_TABLE, HUB_PK),
    "fdc_id_of_sample_food": ("sample_food", HUB_PK),
    "fdc_id_of_input_food": (HUB_TABLE, HUB_PK),
    "nutrient_id": ("nutrient", "id"),
    "measure_unit_id": ("measure_unit", "id"),
    "lab_method_id": ("lab_method", "id"),
    "food_nutrient_id": ("food_nutrient", "id"),
    "food_category_id": ("food_category", "id"),
    "wweia_category_code": ("wweia_food_category", "wweia_food_category"),
    "food_group_id": ("food_category", "id"),
    "food_nutrient_conversion_factor_id": (
        "food_protein_conversion_factor",
        "food_nutrient_conversion_factor_id",
    ),
}


def fetch_tables(cur, schema: str) -> list[str]:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """,
        (schema,),
    )
    return [r[0] for r in cur.fetchall()]


def fetch_columns(cur, schema: str) -> dict[str, list[dict]]:
    cur.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable,
               character_maximum_length, numeric_precision
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
        """,
        (schema,),
    )
    by_table: dict[str, list[dict]] = defaultdict(list)
    for table, col, dtype, nullable, char_max, num_prec in cur.fetchall():
        by_table[table].append(
            {
                "name": col,
                "data_type": dtype,
                "nullable": nullable == "YES",
                "char_max_length": char_max,
                "numeric_precision": num_prec,
            }
        )
    return dict(by_table)


def fetch_primary_keys(cur, schema: str) -> dict[str, list[str]]:
    cur.execute(
        """
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = %s
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY tc.table_name, kcu.ordinal_position
        """,
        (schema,),
    )
    pks: dict[str, list[str]] = defaultdict(list)
    for table, col in cur.fetchall():
        pks[table].append(col)
    return dict(pks)


def fetch_foreign_keys(cur, schema: str) -> list[dict]:
    cur.execute(
        """
        SELECT
            tc.table_name AS from_table,
            kcu.column_name AS from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = %s
          AND tc.constraint_type = 'FOREIGN KEY'
        ORDER BY from_table, from_column
        """,
        (schema,),
    )
    return [
        {
            "from_table": r[0],
            "from_column": r[1],
            "to_table": r[2],
            "to_column": r[3],
            "constraint_name": r[4],
            "source": "foreign_key",
        }
        for r in cur.fetchall()
    ]


def infer_joins(
    schema: str,
    tables: list[str],
    columns: dict[str, list[dict]],
    primary_keys: dict[str, list[str]],
    explicit_fks: list[dict],
) -> list[dict]:
    """Infer joins from column naming and FDC hub patterns."""
    explicit_keys = {
        (fk["from_table"], fk["from_column"]) for fk in explicit_fks
    }
    table_set = set(tables)
    joins: list[dict] = []

    pk_lookup = {t: primary_keys.get(t, []) for t in tables}

    for table, cols in columns.items():
        for col in cols:
            name = col["name"]
            if (table, name) in explicit_keys:
                continue

            # Direct map (fdc_id, nutrient_id, …)
            if name in NAMED_JOINS:
                ref_table, ref_col = NAMED_JOINS[name]
                if ref_table in table_set and table != ref_table:
                    joins.append(
                        {
                            "from_table": table,
                            "from_column": name,
                            "to_table": ref_table,
                            "to_column": ref_col,
                            "join_type": "LEFT",
                            "source": "named_convention",
                            "note": _join_note(table, name, ref_table),
                        }
                    )
                continue

            # {table}_id -> table.id (skip self)
            m = re.fullmatch(r"(.+)_id", name)
            if m:
                stem = m.group(1)
                candidates = [stem, stem.replace("_", "")]
                for ref_table in candidates:
                    if ref_table not in table_set or ref_table == table:
                        continue
                    ref_pk = pk_lookup.get(ref_table) or ["id"]
                    ref_col = ref_pk[0] if len(ref_pk) == 1 else "id"
                    if ref_col in {c["name"] for c in columns[ref_table]}:
                        joins.append(
                            {
                                "from_table": table,
                                "from_column": name,
                                "to_table": ref_table,
                                "to_column": ref_col,
                                "join_type": "LEFT",
                                "source": "suffix_id_heuristic",
                            }
                        )
                        break

    # Deduplicate
    seen: set[tuple] = set()
    unique: list[dict] = []
    for j in joins:
        key = (j["from_table"], j["from_column"], j["to_table"], j["to_column"])
        if key not in seen:
            seen.add(key)
            unique.append(j)
    return sorted(unique, key=lambda x: (x["from_table"], x["from_column"]))


def _join_note(from_table: str, col: str, to_table: str) -> str | None:
    if col == "food_category_id" and to_table == "food_category":
        return (
            "food.food_category_id is often a category label (text) for branded foods, "
            "not always food_category.id"
        )
    if col == "fdc_id" and from_table != HUB_TABLE and to_table == HUB_TABLE:
        return "Extension table: one row per fdc_id (1:1 or 1:N)"
    return None


def build_star_hub_summary(joins: list[dict], tables: list[str]) -> dict:
    """Summarize food.fdc_id as the central hub."""
    satellites = sorted(
        {
            j["from_table"]
            for j in joins
            if j["to_table"] == HUB_TABLE and j["from_column"] in ("fdc_id", "food_id")
        }
    )
    return {
        "hub_table": f"{HUB_TABLE}.{HUB_PK}",
        "satellite_tables_via_fdc_id": satellites,
        "all_tables": tables,
    }


def format_report(payload: dict) -> str:
    lines = [
        f"Schema: {payload['schema']}",
        f"Tables ({len(payload['tables'])}): {', '.join(payload['tables'])}",
        "",
        "=== Hub ===",
        json.dumps(payload["hub_summary"], indent=2),
        "",
        "=== Explicit foreign keys ===",
    ]
    if payload["foreign_keys"]:
        for fk in payload["foreign_keys"]:
            lines.append(
                f"  {fk['from_table']}.{fk['from_column']} -> "
                f"{fk['to_table']}.{fk['to_column']}"
            )
    else:
        lines.append("  (none declared in database)")

    lines.append("")
    lines.append("=== Inferred joins ===")
    for j in payload["inferred_joins"]:
        note = f"  -- {j['note']}" if j.get("note") else ""
        lines.append(
            f"  {j['from_table']}.{j['from_column']} -> "
            f"{j['to_table']}.{j['to_column']} [{j['source']}]{note}"
        )

    lines.append("")
    lines.append("=== Example SQL (nutrients for a food) ===")
    lines.append(
        """\
SELECT f.fdc_id, f.description, n.name AS nutrient, fn.amount
FROM usda.food f
JOIN usda.food_nutrient fn ON fn.fdc_id = f.fdc_id
JOIN usda.nutrient n ON n.id = fn.nutrient_id
WHERE f.fdc_id = 1105904
LIMIT 20;"""
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer USDA schema and joins")
    parser.add_argument("--schema", default="usda", help="Postgres schema name")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "usda_schema_inferred.json",
        help="Write JSON report to this path",
    )
    args = parser.parse_args()
    load_dotenv()

    with connect() as conn:
        with conn.cursor() as cur:
            tables = fetch_tables(cur, args.schema)
            if not tables:
                print(f"No tables in schema '{args.schema}'.", file=sys.stderr)
                sys.exit(1)
            columns = fetch_columns(cur, args.schema)
            primary_keys = fetch_primary_keys(cur, args.schema)
            foreign_keys = fetch_foreign_keys(cur, args.schema)
            inferred = infer_joins(
                args.schema, tables, columns, primary_keys, foreign_keys
            )

    payload = {
        "schema": args.schema,
        "tables": tables,
        "columns": columns,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "inferred_joins": inferred,
        "hub_summary": build_star_hub_summary(inferred, tables),
    }

    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(format_report(payload))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
