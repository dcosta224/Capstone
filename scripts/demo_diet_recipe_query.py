#!/usr/bin/env python3
"""Quick demo: Vietnamese + dairy-free + high-protein/low-carb via diet tags + Qwen.

Uses RecipeNLG rows (no Supabase). Tags ingredients by keywords/FoodOn cache,
filters dairy-free recipes, ranks by semantic fit, asks local Qwen to pick one.

Usage:
  uv run python scripts/demo_diet_recipe_query.py
  uv run python scripts/demo_diet_recipe_query.py --llm-url http://10.0.0.2:1234/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foodon_web"))

from diet_tags_core import load_diet_tags, tag_ingredient, tag_recipe
from foodon_contains_core import load_contains_table

RECIPE_NLG = ROOT / "Data" / "recipes" / "RecipeNLG.csv"

# Hand-picked Vietnamese-ish rows (title substring -> recipe id in file scan)
DEMO_RECIPE_TITLES = (
    "Vietnamese Spring Rolls",
    "Amy Tran'S Vietnamese Soup",
    "Vietnamese Fried Eggs",
    "Vietnamese Rice",
    "Vietnamese Fried Rice",
    "Grilled Lemongrass Chicken",  # may not exist; skipped if missing
    "Pho Bo",  # may not exist
)

QUERY_TEXT = (
    "Vietnamese recipe, no dairy, high protein, low carbohydrate — "
    "something like grilled lemongrass chicken, pho without noodles, or spring rolls with shrimp"
)


def _parse_nlg_ingredients(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]


def _load_vietnamese_recipes(limit: int = 12) -> list[dict]:
    if not RECIPE_NLG.is_file():
        raise FileNotFoundError(f"Missing {RECIPE_NLG}")

    rows: list[dict] = []
    with RECIPE_NLG.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = str(row.get("title") or row.get("Title") or "").strip()
            if not title:
                continue
            lower = title.lower()
            if "vietnamese" not in lower and "pho" not in lower and "lemongrass" not in lower:
                continue
            ing_raw = row.get("ingredients") or row.get("Ingredients") or ""
            ingredients = _parse_nlg_ingredients(ing_raw)
            if not ingredients:
                continue
            rows.append(
                {
                    "recipe_id": int(row.get("id") or len(rows) + 1),
                    "title": title,
                    "ingredients": ingredients,
                    "semantic_text": f"{title}. Ingredients: {', '.join(ingredients[:12])}",
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _tag_recipe(recipe: dict, registry, contains_table) -> dict:
    ing_rows: list[dict] = []
    contains_union: set[str] = set()
    for i, ing in enumerate(recipe["ingredients"]):
        tagged = tag_ingredient(
            recipe["recipe_id"] * 1000 + i,
            ing,
            None,
            registry,
            foodon_contains_table=contains_table,
        )
        row = {
            "contains_set": set(tagged["contains_set"]),
            "tags": tagged["tags"],
        }
        ing_rows.append(row)
        contains_union |= row["contains_set"]

    # Heuristic per-serving macros for demo ranking (not USDA-linked).
    text = " ".join(recipe["ingredients"]).lower()
    protein_g = 8.0
    carbs_g = 35.0
    fat_g = 10.0
    if any(w in text for w in ("chicken", "shrimp", "pork", "beef", "egg", "fish")):
        protein_g += 18.0
    if any(w in text for w in ("rice", "noodle", "bun", "rice paper")):
        carbs_g += 25.0
    if any(w in text for w in ("oil", "butter", "cheese", "cream")):
        fat_g += 8.0
    if "spring roll" in recipe["title"].lower() or "rice paper" in text:
        carbs_g += 10.0
        protein_g += 5.0

    nutrients = {
        "protein_g": protein_g,
        "carbohydrate_g": carbs_g,
        "total_fat_g": fat_g,
        "energy_kcal": protein_g * 4 + carbs_g * 4 + fat_g * 9,
    }

    rolled = tag_recipe(
        recipe["recipe_id"],
        recipe["title"],
        ing_rows,
        registry,
        nutrient_totals_per_serving=nutrients,
    )
    return {
        **recipe,
        "contains_union": sorted(contains_union),
        "recipe_tags": rolled["tags"],
        "nutrients_per_serving": nutrients,
    }


def _semantic_rank(query: str, recipes: list[dict]) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    texts = [r["semantic_text"] for r in recipes]
    r_embs = model.encode(texts, normalize_embeddings=True)
    sims = r_embs @ q_emb
    order = np.argsort(-sims)
    out = []
    for idx in order:
        r = dict(recipes[int(idx)])
        r["semantic_sim"] = float(sims[int(idx)])
        out.append(r)
    return out


def _qwen_pick(
  query: str,
  candidates: list[dict],
  *,
  model: str,
  base_url: str,
) -> dict:
    lines = [
        "Pick the ONE best recipe for the user. Respond JSON only:",
        '{"chosen_recipe_id": <int>, "rationale": "<2 sentences>"}',
        "",
        f"User request: {query}",
        "",
        "Candidates:",
    ]
    for c in candidates:
        tags = {k: v for k, v in c["recipe_tags"].items() if v is not None}
        lines.append(f"\n--- id={c['recipe_id']}: {c['title']} ---")
        lines.append(f"ingredients: {', '.join(c['ingredients'][:14])}")
        lines.append(f"contains: {c['contains_union'] or ['(none)']}")
        lines.append(f"diet tags: {tags}")
        lines.append(
            f"macros (est. per serving): protein={c['nutrients_per_serving']['protein_g']:.0f}g, "
            f"carbs={c['nutrients_per_serving']['carbohydrate_g']:.0f}g"
        )
        lines.append(f"semantic_sim: {c.get('semantic_sim', 0):.3f}")

    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You select recipes matching dietary and cuisine requests. JSON only.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ],
        "temperature": 0,
    }
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.DOTALL)
    return json.loads(m.group(0)) if m else {"raw": content}


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo diet-filtered recipe query")
    parser.add_argument("--llm-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--llm-url", default="http://10.0.0.2:1234/v1")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    registry = load_diet_tags()
    contains_table = load_contains_table()

    recipes = _load_vietnamese_recipes(limit=15)
    if not recipes:
        raise SystemExit("No Vietnamese recipes found in RecipeNLG.csv")

    tagged = [_tag_recipe(r, registry, contains_table) for r in recipes]

    # Hard filter: dairy-free (restriction layer).
    dairy_free = [r for r in tagged if r["recipe_tags"].get("dairy_free") is True]
    print(f"Loaded {len(tagged)} Vietnamese-ish recipes; {len(dairy_free)} dairy-free\n")

    pool = dairy_free if dairy_free else tagged
    ranked = _semantic_rank(QUERY_TEXT, pool)

    # Prefer recipes that also pass high_protein + low_carb tags (heuristic macros).
    def _goal_score(r: dict) -> tuple[int, float]:
        tags = r["recipe_tags"]
        goals = sum(
            1
            for k in ("high_protein", "low_carb")
            if tags.get(k) is True
        )
        return (goals, r.get("semantic_sim", 0.0))

    ranked.sort(key=_goal_score, reverse=True)
    top = ranked[: args.top_k]

    print("Top candidates after dairy-free + semantic/goal ranking:\n")
    for r in top:
        t = r["recipe_tags"]
        n = r["nutrients_per_serving"]
        print(
            f"  [{r['recipe_id']}] {r['title']!r}\n"
            f"    sim={r.get('semantic_sim', 0):.3f} | "
            f"dairy_free={t.get('dairy_free')} high_protein={t.get('high_protein')} "
            f"low_carb={t.get('low_carb')}\n"
            f"    est. protein={n['protein_g']:.0f}g carbs={n['carbohydrate_g']:.0f}g | "
            f"contains={r['contains_union']}\n"
        )

    print("Asking Qwen to choose...\n")
    pick = _qwen_pick(QUERY_TEXT, top, model=args.llm_model, base_url=args.llm_url)
    chosen_id = pick.get("chosen_recipe_id")
    chosen = next((r for r in top if r["recipe_id"] == chosen_id), top[0])
    print("=" * 60)
    print(f"Qwen pick: [{chosen['recipe_id']}] {chosen['title']}")
    print(f"Rationale: {pick.get('rationale', pick)}")
    print(f"Ingredients: {', '.join(chosen['ingredients'][:10])}...")


if __name__ == "__main__":
    main()
