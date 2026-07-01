#!/usr/bin/env python3
"""Use a local LLM to find FoodOn classes our rules may have missed (untagged gaps).

Targets FOODON_* rows with no current contains_* flags that look like real foods,
asks an LLM which restrictions apply, and writes a review queue of suggested additions.

Usage:
  uv run python scripts/build_foodon_contains_cache.py
  uv run python scripts/audit_foodon_contains_llm.py --candidates-only --limit 30
  uv run python scripts/audit_foodon_contains_llm.py --limit 200 --batch-size 8 \\
    --llm-model qwen/qwen3.6-35b-a3b --llm-url http://10.0.0.2:1234/v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from foodon_contains_core import load_contains_table, lookup_contains
from foodon_contains_judge import FoodOnContainsLLMJudge
from foodon_contains_ontology import (
    evaluate_llm_suggestions,
    format_ontology_summary,
)
from foodon_index import FoodOnIndex
from foodon_paths import FOODON_CONTAINS_CSV, FOODON_INDEX_CACHE

OUT_DIR = ROOT / "scratch" / "tag"
DEFAULT_OUT = OUT_DIR / "foodon_contains_llm_review.csv"
DEFAULT_CANDIDATES = OUT_DIR / "foodon_contains_llm_candidates.csv"
DEFAULT_SUMMARY = OUT_DIR / "foodon_contains_llm_summary.json"

SKIP_LABEL_RE = re.compile(
    r"\b(?:process|specification|obsolete|category|entity|method|quality|attribute|"
    r"evaluation|reference|index|annotation|datum|byproduct fines)\b",
    re.IGNORECASE,
)

FOODISH_LABEL_RE = re.compile(
    r"\b(?:food product|beverage|meat|stew|hash|broth|soup|cheese|milk|cream|"
    r"fish|salmon|tuna|shrimp|crab|lobster|egg|honey|wine|beer|flour|bread|"
    r"pasta|tofu|sausage|bacon|ham|pork|beef|lamb|veal|chicken|turkey|duck|"
    r"almond|walnut|peanut|sesame|soy|bean curd|yogurt|butter|oil)\b",
    re.IGNORECASE,
)

HINT_SCORE_WORDS = (
    "meat",
    "poultry",
    "chicken",
    "turkey",
    "duck",
    "beef",
    "lamb",
    "veal",
    "pork",
    "swine",
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "crab",
    "lobster",
    "shellfish",
    "milk",
    "cheese",
    "cream",
    "butter",
    "yogurt",
    "egg",
    "wheat",
    "flour",
    "bread",
    "peanut",
    "almond",
    "walnut",
    "cashew",
    "soy",
    "tofu",
    "honey",
    "wine",
    "beer",
    "sausage",
    "bacon",
    "onion",
    "garlic",
    "potato",
)


def _hint_score(label: str) -> int:
    text = label.lower()
    return sum(1 for w in HINT_SCORE_WORDS if w in text)


def select_untagged_candidates(table: pd.DataFrame, *, limit: int | None) -> pd.DataFrame:
    cols = [c for c in table.columns if c.startswith("contains_")]
    tagged_any = table[cols].astype(bool).any(axis=1)
    untagged = table[~tagged_any].copy()
    untagged = untagged[untagged["foodon_id"].astype(str).str.startswith("FOODON_")]

    labels = untagged["label"].astype(str)
    mask_skip = ~labels.str.contains(SKIP_LABEL_RE, na=False)
    mask_food = labels.str.contains(FOODISH_LABEL_RE, na=False)
    untagged = untagged[mask_skip & mask_food]
    untagged = untagged.copy()
    untagged["hint_score"] = untagged["label"].astype(str).map(_hint_score)
    untagged = untagged.sort_values(["hint_score", "label"], ascending=[False, True])
    if limit is not None:
        untagged = untagged.head(limit)
    return untagged


def _load_checkpoint(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = pd.read_csv(path, usecols=["foodon_id"])
    return set(done["foodon_id"].astype(str).tolist())


def _nonempty_col(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.len() > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM audit for untagged FoodOn contains gaps")
    parser.add_argument("--limit", type=int, default=100, help="Max candidates to judge")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument(
        "--min-ontology-score",
        type=float,
        default=0.55,
        help="Minimum ontology score to confirm an LLM suggestion",
    )
    parser.add_argument("--candidates-only", action="store_true", help="List candidates, no LLM")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidates-out", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:1234/v1")
    parser.add_argument("--llm-api", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--ollama-model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ollama-url", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    llm_model = args.llm_model or args.ollama_model
    llm_url = args.ollama_url or args.llm_url
    if args.ollama_model and not args.llm_model:
        args.llm_api = "ollama"

    table = load_contains_table()
    if table is None:
        print(f"Missing contains table. Run build_foodon_contains_cache.py", flush=True)
        raise SystemExit(1)

    candidates = select_untagged_candidates(table, limit=args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates[["foodon_id", "label", "hint_score"]].to_csv(args.candidates_out, index=False)
    print(f"Untagged food-like candidates: {len(candidates):,}", flush=True)
    print(f"Wrote {args.candidates_out}", flush=True)

    if args.candidates_only or not len(candidates):
        if not args.candidates_only and not len(candidates):
            print("No candidates to judge.", flush=True)
        raise SystemExit(0)

    if not llm_model:
        print(
            "Provide --llm-model and --llm-url (or use --candidates-only).",
            flush=True,
        )
        raise SystemExit(1)

    if not FOODON_INDEX_CACHE.is_file():
        print("Run: uv run python scripts/build_foodon_index_cache.py", flush=True)
        raise SystemExit(1)
    foodon_index = FoodOnIndex.from_cache(FOODON_INDEX_CACHE)

    judge = FoodOnContainsLLMJudge(model=llm_model, base_url=llm_url, api=args.llm_api)
    done_ids = _load_checkpoint(args.out)
    pending = candidates[~candidates["foodon_id"].astype(str).isin(done_ids)]

    print(
        f"LLM judge: {llm_model} @ {llm_url} ({args.llm_api}) — {len(pending):,} to judge",
        flush=True,
    )

    rows: list[dict] = []
    judged = 0
    t0 = time.perf_counter()

    def _ontology_context_line(node_id: str) -> str:
        parents = foodon_index.parents.get(node_id, [])[:2]
        parent_bits = [
            f"{foodon_index.labels.get(p, p)} [{','.join(sorted(lookup_contains(p, table))[:3]) or 'none'}]"
            for p in parents
        ]
        siblings = []
        if parents:
            for sib in foodon_index.children.get(parents[0], [])[:6]:
                if sib == node_id:
                    continue
                sib_tags = sorted(lookup_contains(sib, table))
                if sib_tags:
                    siblings.append(f"{foodon_index.labels.get(sib, sib)}={','.join(sib_tags[:2])}")
        parts = []
        if parent_bits:
            parts.append("parents: " + "; ".join(parent_bits))
        if siblings:
            parts.append("tagged siblings: " + "; ".join(siblings[:3]))
        return " | ".join(parts)

    def _process_results(results: list[dict]) -> None:
        batch_tags = {
            res["foodon_id"]: set(res.get("contains") or [])
            for res in results
            if not res.get("error")
        }
        for res in results:
            node_id = res["foodon_id"]
            current = set(lookup_contains(node_id, table))
            llm = set(res.get("contains") or [])
            added = sorted(llm - current)
            conf = float(res.get("confidence") or 0.0)
            confirmed, rejected, verdicts = evaluate_llm_suggestions(
                node_id=node_id,
                label=res["label"],
                current=current,
                llm_contains=llm,
                llm_confidence=conf,
                foodon_index=foodon_index,
                contains_table=table,
                llm_batch_tags=batch_tags,
                min_confirm_score=args.min_ontology_score,
            )
            rows.append(
                {
                    "foodon_id": node_id,
                    "label": res["label"],
                    "current_contains": ",".join(sorted(current)),
                    "llm_contains": ",".join(sorted(llm)),
                    "llm_added": ",".join(added),
                    "ontology_confirmed": ",".join(confirmed),
                    "ontology_rejected": ",".join(rejected),
                    "ontology_summary": format_ontology_summary(verdicts),
                    "llm_confidence": conf,
                    "llm_rationale": res.get("rationale", ""),
                    "llm_error": bool(res.get("error")),
                }
            )

    batch_items: list[dict[str, str]] = []
    for row in pending.itertuples(index=False):
        node_id = str(row.foodon_id)
        batch_items.append(
            {
                "foodon_id": node_id,
                "label": str(row.label),
                "ontology_context": _ontology_context_line(node_id),
            }
        )
        if len(batch_items) < args.batch_size:
            continue

        _process_results(judge.classify_batch(batch_items))
        judged += len(batch_items)
        batch_items = []

        if judged % max(args.checkpoint_every, args.batch_size) == 0:
            _append_rows(args.out, rows)
            rows = []
            print(f"  judged {judged:,} ({time.perf_counter() - t0:.0f}s)", flush=True)

    if batch_items:
        _process_results(judge.classify_batch(batch_items))
        judged += len(batch_items)

    _append_rows(args.out, rows)

    review = pd.read_csv(args.out) if args.out.is_file() else pd.DataFrame()
    if len(review) and args.min_confidence:
        flagged = review[
            _nonempty_col(review["ontology_confirmed"])
            & (review["llm_confidence"].astype(float) >= args.min_confidence)
            & (~review["llm_error"].astype(bool))
        ]
        raw_flagged = review[
            _nonempty_col(review["llm_added"])
            & (review["llm_confidence"].astype(float) >= args.min_confidence)
            & (~review["llm_error"].astype(bool))
        ]
    else:
        flagged = review.iloc[0:0]
        raw_flagged = review.iloc[0:0]
    summary = {
        "candidates": len(candidates),
        "judged": judged,
        "review_rows": len(review),
        "with_llm_additions": int(_nonempty_col(raw_flagged["llm_added"]).sum())
        if len(raw_flagged)
        else 0,
        "with_ontology_confirmed": int(_nonempty_col(flagged["ontology_confirmed"]).sum())
        if len(flagged)
        else 0,
        "llm_model": llm_model,
        "llm_url": llm_url,
        "min_confidence": args.min_confidence,
        "min_ontology_score": args.min_ontology_score,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.out}", flush=True)
    if len(flagged):
        print("\nSample ontology-confirmed additions:", flush=True)
        for r in flagged.head(10).itertuples(index=False):
            print(
                f"  [{r.llm_confidence:.2f}] {r.label!r} [{r.foodon_id}]: "
                f"+{r.ontology_confirmed} ({r.ontology_summary}) — {r.llm_rationale}",
                flush=True,
            )
    if len(raw_flagged) > len(flagged):
        print(
            f"\nFiltered {len(raw_flagged) - len(flagged)} raw LLM suggestions without ontology support.",
            flush=True,
        )


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.is_file()
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
