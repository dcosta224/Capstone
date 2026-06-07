#!/usr/bin/env python3
"""Funnel chart: non-branded USDA foods → dedup by description → 4 macros."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingredient_match import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "All_Food_Data_April_2026"
DEFAULT_OUT_HTML = ROOT / "scratch" / "food_4macro_funnel.html"
DEFAULT_OUT_PNG = ROOT / "scratch" / "food_4macro_funnel.png"

BRANDED_TYPE = "branded_food"
MACRO_NUTRIENT_IDS = frozenset({1003, 1004, 1005, 1008})  # protein, fat, carb, energy

STAGE_LABELS = [
    "Non-branded foods",
    "Deduped by description",
    "All 4 core macros",
]
STAGE_RULES = [
    "usda.food where data_type ≠ branded_food",
    "one row per normalized description (lowercased, punctuation stripped); latest publication_date wins",
    "canonical deduped rows with nutrient_id 1003, 1004, 1005, and 1008 in food_nutrient",
]

# Plotly textposition per stage (dedup label: outside right, dark text)
STAGE_TEXT_POSITIONS = ["inside", "outside", "outside"]
# Prior-stage label for % retained (index 0 has no prior stage)
STAGE_PCT_OF_PRIOR = [None, "Non-branded", "deduped"]


def normalize_food_description(description: str) -> str:
    """Same rules as sql/06_create_food_mvp.sql normalize_food_description / ingredient_match.normalize_text."""
    return normalize_text(description)


def count_all_food_from_csv() -> int:
    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id"],
        dtype={"fdc_id": "int64"},
    )
    return len(food)


def count_all_food_from_csv() -> int:
    food = pd.read_csv(DATA_DIR / "food.csv", usecols=["fdc_id"], dtype={"fdc_id": "int64"})
    return len(food)


def count_non_branded_from_csv() -> int:
    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id", "data_type"],
        dtype={"fdc_id": "int64", "data_type": "string"},
    )
    return int((food["data_type"] != BRANDED_TYPE).sum())


def non_branded_fdc_ids_from_csv() -> set[int]:
    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id", "data_type"],
        dtype={"fdc_id": "int64", "data_type": "string"},
    )
    return set(food.loc[food["data_type"] != BRANDED_TYPE, "fdc_id"])


def four_macro_fdc_ids_non_branded_from_csv(*, chunk_size: int = 1_000_000) -> set[int]:
    universe = non_branded_fdc_ids_from_csv()
    tallies: dict[int, set[int]] = defaultdict(set)
    path = DATA_DIR / "food_nutrient.csv"
    for chunk in pd.read_csv(path, usecols=["fdc_id", "nutrient_id"], chunksize=chunk_size):
        chunk = chunk[chunk["fdc_id"].isin(universe) & chunk["nutrient_id"].isin(MACRO_NUTRIENT_IDS)]
        if chunk.empty:
            continue
        for fdc_id, nutrients in chunk.groupby("fdc_id")["nutrient_id"]:
            tallies[int(fdc_id)].update(int(n) for n in nutrients.unique())
    return {fdc_id for fdc_id, nutrients in tallies.items() if nutrients >= MACRO_NUTRIENT_IDS}


def deduped_non_branded_fdc_ids_from_csv() -> set[int]:
    """One canonical fdc_id per normalized description among all non-branded foods."""
    universe = non_branded_fdc_ids_from_csv()
    food = pd.read_csv(
        DATA_DIR / "food.csv",
        usecols=["fdc_id", "description", "publication_date", "data_type"],
        dtype={"fdc_id": "int64", "description": "string", "publication_date": "string", "data_type": "string"},
    )
    eligible = food[food["fdc_id"].isin(universe)].copy()
    eligible["norm"] = eligible["description"].fillna("").map(normalize_food_description)
    empty = eligible["norm"] == ""
    eligible.loc[empty, "norm"] = "fdc_id:" + eligible.loc[empty, "fdc_id"].astype(str)
    eligible = eligible.sort_values(
        ["norm", "publication_date", "fdc_id"],
        ascending=[True, False, False],
        na_position="last",
    )
    canonical = eligible.drop_duplicates(subset=["norm"], keep="first")
    return set(canonical["fdc_id"].astype(int))


def count_all_food_from_db() -> int:
    from db import connect

    with connect() as conn:
        row = pd.read_sql("SELECT COUNT(*) AS n FROM usda.food", conn)
    return int(row["n"].iloc[0])


def count_non_branded_from_db() -> int:
    from db import connect

    sql = "SELECT COUNT(*) AS n FROM usda.food WHERE data_type <> %s"
    with connect() as conn:
        row = pd.read_sql(sql, conn, params=(BRANDED_TYPE,))
    return int(row["n"].iloc[0])


def count_funnel_stages_from_db() -> dict[str, int]:
    """Non-branded → dedup → 4 macros (funnel order)."""
    from db import connect

    sql = """
    WITH required_macros (nutrient_id) AS (
        VALUES (1003), (1004), (1005), (1008)
    ),
    fdc_with_all_macros AS (
        SELECT fn.fdc_id
        FROM usda.food_nutrient fn
        INNER JOIN required_macros rm ON rm.nutrient_id = fn.nutrient_id
        GROUP BY fn.fdc_id
        HAVING COUNT(DISTINCT fn.nutrient_id) = 4
    ),
    non_branded AS (
        SELECT f.fdc_id, f.description, f.publication_date
        FROM usda.food f
        WHERE f.data_type <> %s
    ),
    deduped AS (
        SELECT
            nb.fdc_id,
            ROW_NUMBER() OVER (
                PARTITION BY COALESCE(
                    NULLIF(usda.normalize_food_description(nb.description), ''),
                    'fdc_id:' || nb.fdc_id::text
                )
                ORDER BY nb.publication_date DESC NULLS LAST, nb.fdc_id DESC
            ) AS dedupe_rank
        FROM non_branded nb
    ),
    canonical AS (
        SELECT fdc_id FROM deduped WHERE dedupe_rank = 1
    )
    SELECT
        (SELECT COUNT(*) FROM usda.food) AS total_food,
        (SELECT COUNT(*) FROM non_branded) AS non_branded,
        (SELECT COUNT(*) FROM canonical) AS deduped,
        (SELECT COUNT(*) FROM canonical c INNER JOIN fdc_with_all_macros m ON m.fdc_id = c.fdc_id) AS four_macro
    """
    with connect() as conn:
        row = pd.read_sql(sql, conn, params=(BRANDED_TYPE,))
    return {
        "total_food": int(row["total_food"].iloc[0]),
        "non_branded": int(row["non_branded"].iloc[0]),
        "deduped": int(row["deduped"].iloc[0]),
        "four_macro": int(row["four_macro"].iloc[0]),
    }


def compute_counts(*, source: str) -> dict[str, int]:
    if source == "csv":
        if not (DATA_DIR / "food.csv").is_file():
            raise FileNotFoundError(f"Missing {DATA_DIR / 'food.csv'}")
        if not (DATA_DIR / "food_nutrient.csv").is_file():
            raise FileNotFoundError(f"Missing {DATA_DIR / 'food_nutrient.csv'}")
        total_food = count_all_food_from_csv()
        print(f"All usda.food rows: {total_food:,}", flush=True)
        non_branded = count_non_branded_from_csv()
        print(f"Non-branded foods: {non_branded:,}", flush=True)
        canonical_ids = deduped_non_branded_fdc_ids_from_csv()
        deduped = len(canonical_ids)
        print(f"Deduped by description: {deduped:,}", flush=True)
        four_macro_ids = four_macro_fdc_ids_non_branded_from_csv()
        four_macro = len(canonical_ids & four_macro_ids)
        print(f"All 4 core macros (after dedup): {four_macro:,}", flush=True)
    elif source == "db":
        counts = count_funnel_stages_from_db()
        print(f"All usda.food rows: {counts['total_food']:,}", flush=True)
        print(f"Non-branded foods: {counts['non_branded']:,}", flush=True)
        print(f"Deduped by description: {counts['deduped']:,}", flush=True)
        print(f"All 4 core macros (after dedup): {counts['four_macro']:,}", flush=True)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    if source == "csv":
        return {
            "total_food": total_food,
            "non_branded": non_branded,
            "deduped": deduped,
            "four_macro": four_macro,
        }
    return counts


def build_funnel_dataframe(counts: dict[str, int]) -> pd.DataFrame:
    total_food = counts["total_food"]
    funnel = pd.DataFrame(
        {
            "stage": STAGE_LABELS,
            "rule": STAGE_RULES,
            "count": [
                counts["non_branded"],
                counts["deduped"],
                counts["four_macro"],
            ],
        }
    )
    funnel["pct_of_all_foods"] = (100 * funnel["count"] / total_food).round(2)
    funnel["pct_of_previous"] = (100 * funnel["count"] / funnel["count"].shift(1)).round(2)
    funnel["dropped_from_previous"] = (
        funnel["count"].shift(1) - funnel["count"]
    ).fillna(0).astype(int)
    return funnel


def _segment_pct_html(row: Any, stage_idx: int, *, sub_color: str) -> str:
    """Stage 1: % of all USDA only; stages 2–3: % of prior stage, then % of all USDA (bottom)."""
    all_usda = f"{row.pct_of_all_foods:.2f}% of all USDA foods"
    wrap = f"<span style='font-size:11px;color:{sub_color}'>"
    if stage_idx == 0:
        return f"{wrap}{all_usda}</span>"
    prior_name = STAGE_PCT_OF_PRIOR[stage_idx]
    prior_line = f"{row.pct_of_previous:.2f}% of {prior_name}"
    return f"{wrap}{prior_line}<br>{all_usda}</span>"


def plot_funnel(funnel: pd.DataFrame, *, out_html: Path, out_png: Path | None) -> go.Figure:
    FONT_FAMILY = "Inter, system-ui, sans-serif"
    FUNNEL_COLORS = ["#1d4ed8", "#3b82f6", "#059669"]
    FUNNEL_BORDERS = ["#93c5fd", "#bae6fd", "#6ee7b7"]
    INSIDE_COLORS = ("#ffffff", "#f1f5f9", "#bfdbfe")
    OUTSIDE_COLORS = ("#0f172a", "#0f172a", "#334155")

    n = len(funnel)
    text_positions = STAGE_TEXT_POSITIONS[:n]
    hover_prev_labels = [
        ""
        if i == 0 or pd.isna(row.pct_of_previous) or STAGE_PCT_OF_PRIOR[i] is None
        else f"{row.pct_of_previous:.2f}% of {STAGE_PCT_OF_PRIOR[i]}<br>"
        for i, row in enumerate(funnel.itertuples())
    ]
    funnel_plot = funnel.assign(hover_prev_label=hover_prev_labels)

    segment_text = []
    for stage_idx, (row, pos) in enumerate(zip(funnel_plot.itertuples(), text_positions, strict=True)):
        title_c, value_c, sub_c = INSIDE_COLORS if pos == "inside" else OUTSIDE_COLORS
        segment_text.append(
            f"<b style='color:{title_c}'>{row.stage}</b><br>"
            f"<span style='color:{value_c}'>{int(row.count):,}</span><br>"
            f"{_segment_pct_html(row, stage_idx, sub_color=sub_c)}"
        )

    label_font = dict(size=14, color=INSIDE_COLORS[0], family=FONT_FAMILY)
    outside_font = dict(size=13, color=OUTSIDE_COLORS[0], family=FONT_FAMILY)

    fig = go.Figure(
        go.Funnel(
            y=funnel_plot["stage"],
            x=funnel_plot["count"],
            text=segment_text,
            textinfo="text",
            customdata=funnel_plot[
                ["rule", "dropped_from_previous", "pct_of_all_foods", "hover_prev_label"]
            ].values,
            textposition=text_positions,
            insidetextfont=label_font,
            outsidetextfont=outside_font,
            cliponaxis=False,
            constraintext="none",
            hovertemplate=(
                "<b>%{label}</b><extra></extra><br>"
                "Filter: %{customdata[0]}<br>"
                "Foods: <b>%{value:,}</b><br>"
                "%{customdata[2]:.2f}% of all USDA foods<br>"
                "%{customdata[3]}"
                "Dropped from prior: %{customdata[1]:,}"
            ),
            marker=dict(
                color=FUNNEL_COLORS,
                line=dict(color=FUNNEL_BORDERS, width=[2] * n),
            ),
            connector=dict(
                line=dict(color="rgba(148, 163, 184, 0.55)", width=1.5),
                fillcolor="rgba(226, 232, 240, 0.35)",
            ),
            opacity=0.95,
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>MVP Data Cleaning Funnel</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18, family=FONT_FAMILY, color="#0f172a"),
        ),
        funnelmode="stack",
        template="plotly_white",
        width=920,
        height=560,
        margin=dict(l=40, r=200, t=72, b=32),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        font=dict(family=FONT_FAMILY, color="#334155", size=12),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cbd5e1",
            font=dict(family=FONT_FAMILY, size=12),
        ),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}", flush=True)

    if out_png is not None:
        try:
            fig.write_image(out_png, width=920, height=480, scale=2)
            print(f"Wrote {out_png}", flush=True)
        except Exception as exc:
            print(f"PNG export skipped ({exc}). Install kaleido: uv add kaleido", flush=True)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot non-branded → 4-macro food funnel.")
    parser.add_argument(
        "--source",
        choices=("csv", "db"),
        default="csv",
        help="Count from local April 2026 CSVs (default) or Supabase",
    )
    parser.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write stage counts as JSON",
    )
    args = parser.parse_args()

    counts = compute_counts(source=args.source)
    funnel = build_funnel_dataframe(counts)
    print()
    print(funnel.to_string(index=False))

    plot_funnel(
        funnel,
        out_html=args.out_html,
        out_png=None if args.no_png else args.out_png,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stages": funnel.to_dict(orient="records"),
            "source": args.source,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
