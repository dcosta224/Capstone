"""Analyze contains_* false positives for parent agent report."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\winni\Documents\GitHub\Capstone\Capstone")
sys.path.insert(0, str(ROOT / "scripts"))

from diet_tags_core import detect_contains, load_diet_tags

SLUGS = [
    "alcohol", "dairy", "egg", "fish", "honey", "peanut", "pork", "poultry",
    "red_meat", "root_vegetable", "sesame", "shellfish", "soy", "tree_nut", "wheat",
]
COL = {s: f"contains_{s}" for s in SLUGS}

foodon_path = ROOT / "scratch" / "tag" / "foodon_contains.csv"
review_path = ROOT / "scratch" / "tag" / "ingredient_tag_review.csv"
mapping_path = ROOT / "scratch" / "tag" / "fdc_foodon_mapping.csv"

registry = load_diet_tags()
df = pd.read_csv(foodon_path, low_memory=False)
df["label_lc"] = df["label"].astype(str).str.lower()

# --- FoodOn suspicious heuristics per dimension ---
HEURISTICS: list[tuple[str, str, str, str]] = [
    ("dairy", "contains_dairy == True", "label_lc.str.contains('coconut|almond|oat|soy|rice|plant|non-dairy|dairy-free|vegan|nut milk|685,regex=True)", "plant/non-dairy name tagged dairy (FoodOn propagation)"),
    ("dairy", "contains_dairy == True", "label_lc.str.contains('butter bean|butternut|peanut butter|shea butter|cocoa butter', regex=True)", "butter substring / non-dairy butter (ontology)"),
    ("tree_nut", "contains_tree_nut == True", "label_lc.str.contains('coconut|peanut|nutmeg|nutrient|doughnut|nut shell', regex=True)", "not a tree nut but under nut food product"),
    ("tree_nut", "contains_tree_nut == True", "label_lc.str.contains('chestnut|water chestnut', regex=True)", "chestnut ambiguity"),
    ("peanut", "contains_peanut == True", "label_lc.str.contains('peanut butter cup|groundnut', regex=True)", "peanut lineage edge cases"),
    ("wheat", "contains_wheat == True", "label_lc.str.contains('buckwheat|wheatgrass|wheat germ oil|rice|corn|oat|barley rye', regex=True)", "non-wheat grain or wheatgrass tagged wheat"),
    ("poultry", "contains_poultry == True", "label_lc.str.contains('turkey|chicken|duck|fowl', regex=True) & label_lc.str.contains('broth|stock|soup|gravy', regex=True)", "broth/soup poultry (may be OK) — find non-meat"),
    ("poultry", "contains_poultry == True", "~label_lc.str.contains('chicken|turkey|duck|goose|quail|poultry|fowl|hen|rooster', regex=True)", "poultry tag without obvious poultry in label"),
    ("red_meat", "contains_red_meat == True", "label_lc.str.contains('beef broth|beef stock|beef flavor|beef extract', regex=True)", "flavor/broth red_meat propagation"),
    ("red_meat", "contains_red_meat == True", "~label_lc.str.contains('beef|cattle|veal|lamb|mutton|bison|steak|burger|goat meat|game', regex=True)", "red_meat without obvious red meat label"),
    ("pork", "contains_pork == True", "~label_lc.str.contains('pork|swine|ham|bacon|prosciutto|sausage|pig', regex=True)", "pork tag without pork keywords in label"),
    ("fish", "contains_fish == True", "label_lc.str.contains('goldfish|starfish|cuttlefish', regex=True)", "fish word but not food fish"),
    ("shellfish", "contains_shellfish == True", "label_lc.str.contains('shell|crustacean', regex=True) & ~label_lc.str.contains('shrimp|crab|lobster|clam|mussel|oyster|scallop|crayfish|prawn', regex=True)", "shellfish tag on vague shell terms"),
    ("egg", "contains_egg == True", "label_lc.str.contains('eggplant|egg roll wrapper|vegan egg|egg-free', regex=True)", "egg false positive from name"),
    ("soy", "contains_soy == True", "label_lc.str.contains('soy sauce|soybean|tofu|edamame|miso|tempeh', regex=True)", "reference: soy tagged (sample)"),
    ("soy", "contains_soy == True", "~label_lc.str.contains('soy|soya|tofu|edamame|miso|tempeh', regex=True)", "soy tag without soy in label"),
    ("root_vegetable", "contains_root_vegetable == True", "label_lc.str.contains('ginger|turmeric|horseradish|wasabi', regex=True)", "spice roots tagged root_vegetable"),
    ("root_vegetable", "contains_root_vegetable == True", "label_lc.str.contains('carrot|potato|onion|garlic|beet|turnip|parsnip|rutabaga|yam|sweet potato', regex=True)", "reference root veg"),
    ("honey", "contains_honey == True", "label_lc.str.contains('honeydew|honeysuckle|honey mustard', regex=True)", "honey substring false positives"),
    ("alcohol", "contains_alcohol == True", "label_lc.str.contains('non-alcoholic|alcohol-free|vanilla extract|wine vinegar', regex=True)", "alcohol tag on low/no alcohol products"),
    ("sesame", "contains_sesame == True", "label_lc.str.contains('sesame', regex=True)", "sesame tagged samples"),
]

foodon_fps: dict[str, list[dict]] = {s: [] for s in SLUGS}
seen_ids: dict[str, set] = {s: set() for s in SLUGS}

for slug, base_query, sus_query, why in HEURISTICS:
    if slug not in foodon_fps:
        continue
    try:
        base = df.query(base_query, engine="python")
        sus = base.query(sus_query, engine="python")
    except Exception as e:
        print(f"HEURISTIC FAIL {slug}: {e}", file=sys.stderr)
        continue
    for _, row in sus.head(8).iterrows():
        fid = row["foodon_id"]
        if fid in seen_ids[slug]:
            continue
        seen_ids[slug].add(fid)
        foodon_fps[slug].append({
            "foodon_id": fid,
            "label": row["label"],
            "wrong_tag": slug,
            "why": why,
            "source_type": "FoodOn propagation (ancestor inheritance in foodon_contains.csv)",
        })
        if len(foodon_fps[slug]) >= 4:
            break

# Extra targeted FoodOn searches
TARGETS = [
    ("dairy", df["contains_dairy"] & df["label_lc"].str.contains("coconut milk", na=False)),
    ("tree_nut", df["contains_tree_nut"] & df["label_lc"].str.contains("coconut", na=False)),
    ("wheat", df["contains_wheat"] & df["label_lc"].str.contains("buckwheat", na=False)),
    ("poultry", df["contains_poultry"] & df["label_lc"].str.contains("chicken stew", na=False)),
    ("poultry", df["contains_poultry"] & df["label_lc"].str.contains("chicken soup", na=False)),
    ("red_meat", df["contains_red_meat"] & df["label_lc"].str.contains("cattle soup", na=False)),
]

for slug, mask in TARGETS:
    sub = df[mask].head(3)
    for _, row in sub.iterrows():
        fid = row["foodon_id"]
        if fid in seen_ids[slug] or len(foodon_fps[slug]) >= 4:
            continue
        seen_ids[slug].add(fid)
        foodon_fps[slug].append({
            "foodon_id": fid,
            "label": row["label"],
            "wrong_tag": slug,
            "why": "targeted keyword search on tagged row",
            "source_type": "FoodOn propagation",
        })

# --- Keyword false positives via detect_contains ---
KEYWORD_SAMPLES = [
    ("pepper universal vs black pepper", "black pepper", None),
    ("pepper universal vs black pepper", "bell pepper", None),
    ("pepper universal vs black pepper", "pepper", None),
    ("butter in peanut butter", "peanut butter", None),
    ("butter in peanut butter", "peanut butter sandwich", None),
    ("butter dairy", "butter", None),
    ("ham in hamburger", "hamburger", None),
    ("ham in chatham", "chatham", None),
    ("ham real", "ham sandwich", None),
    ("cream unrelated", "ice cream", None),
    ("cream unrelated", "cream of mushroom soup", None),
    ("cream unrelated", "sunscreen cream", None),
    ("cream unrelated", "cream cheese", None),
    ("nut in doughnut", "doughnut", None),
    ("nut in doughnut", "donut glazed", None),
    ("nut real", "walnut", None),
    ("eggplant", "eggplant parmesan", None),
    ("wine vinegar", "red wine vinegar", None),
    ("soy in soy sauce", "soy sauce", None),
    ("chicken broth", "chicken broth low sodium", None),
]

keyword_fps: list[dict] = []
for case_name, desc, ing in KEYWORD_SAMPLES:
    hits = detect_contains(desc, ing, registry)
    keyword_fps.append({
        "case": case_name,
        "description": desc,
        "hits": {k: v for k, v in hits.items()},
    })

# --- ingredient_tag_review.csv ---
review_samples: dict[str, dict] = {}
if review_path.exists():
    rev = pd.read_csv(review_path)
    for col in ["cache_only_adds", "keyword_only_adds"]:
        rev[col] = rev[col].fillna("").astype(str)
    major_slugs = ["dairy", "fish", "poultry", "wheat", "tree_nut", "pork", "shellfish", "root_vegetable"]
    for col in ["cache_only_adds", "keyword_only_adds"]:
        samples = []
        for slug in major_slugs:
            pat = rf"(^|,){slug}(,|$)"
            m = rev[rev[col].str.contains(pat, regex=True, na=False)]
            for _, row in m.head(10).iterrows():
                samples.append({
                    "fdc_id": int(row["fdc_id"]),
                    "description": row["description"],
                    "foodon_id": row.get("foodon_id"),
                    "foodon_label": row.get("foodon_label"),
                    "confidence": row.get("confidence"),
                    "bucket": row.get("bucket"),
                    col: row[col],
                    "slug": slug,
                })
        review_samples[col] = samples[:80]

# --- mapping disagreements (low confidence + cache vs keyword) ---
mapping_disagreements: list[dict] = []
if mapping_path.exists():
    # review file has the disagreement columns; full mapping may be huge
    rev = pd.read_csv(review_path) if review_path.exists() else None
    if rev is not None:
        rev["confidence"] = pd.to_numeric(rev["confidence"], errors="coerce")
        low = rev[rev["confidence"] < 0.65].copy()
        for _, row in low.head(30).iterrows():
            kw = str(row.get("keyword_contains") or "")
            cache = str(row.get("cache_contains") or "")
            if kw != cache or (row.get("cache_only_adds") or row.get("keyword_only_adds")):
                mapping_disagreements.append({
                    "fdc_id": int(row["fdc_id"]),
                    "description": row["description"],
                    "foodon_id": row.get("foodon_id"),
                    "foodon_label": row.get("foodon_label"),
                    "confidence": float(row["confidence"]),
                    "bucket": row.get("bucket"),
                    "keyword_contains": kw,
                    "cache_contains": cache,
                    "cache_only_adds": row.get("cache_only_adds"),
                    "keyword_only_adds": row.get("keyword_only_adds"),
                    "source_type": "bad partner mapping (low-confidence FoodOn link + contains mismatch)",
                })

# Fill gaps: scan foodon for any dimension with <2 examples using broad negative keyword search
FILL = {
    "fish": ("contains_fish == True", "label_lc.str.contains('fish oil supplement|pet food|aquarium', regex=True)"),
    "honey": ("contains_honey == True", "label_lc.str.contains('honeydew|honeysuckle', regex=True)"),
    "alcohol": ("contains_alcohol == True", "label_lc.str.contains('extract|vinegar|non-alcoholic', regex=True)"),
    "sesame": ("contains_sesame == True", "label_lc.str.contains('sesame oil|sesame seed|tahini|sesame', regex=True)"),
    "shellfish": ("contains_shellfish == True", "label_lc.str.contains('krill|coral', regex=True)"),
    "egg": ("contains_egg == True", "label_lc.str.contains('eggplant', regex=True)"),
    "peanut": ("contains_peanut == True", "label_lc.str.contains('peanut oil|peanut butter', regex=True)"),
}
for slug, (bq, sq) in FILL.items():
    while len(foodon_fps[slug]) < 2:
        try:
            sus = df.query(bq, engine="python").query(sq, engine="python")
        except Exception:
            break
        added = False
        for _, row in sus.iterrows():
            fid = row["foodon_id"]
            if fid in seen_ids[slug]:
                continue
            seen_ids[slug].add(fid)
            foodon_fps[slug].append({
                "foodon_id": fid,
                "label": row["label"],
                "wrong_tag": slug,
                "why": f"fill heuristic: {sq}",
                "source_type": "FoodOn propagation",
            })
            added = True
            break
        if not added:
            # grab any tagged row with suspicious substring from label
            col = COL[slug]
            tagged = df[df[col] == True].head(5)
            for _, row in tagged.iterrows():
                fid = row["foodon_id"]
                if fid not in seen_ids[slug]:
                    seen_ids[slug].add(fid)
                    foodon_fps[slug].append({
                        "foodon_id": fid,
                        "label": row["label"],
                        "wrong_tag": slug,
                        "why": "example tagged node (verify manually)",
                        "source_type": "FoodOn propagation",
                    })
                    break
            break

report = {
    "foodon_false_positives_by_dimension": foodon_fps,
    "keyword_detect_contains_samples": keyword_fps,
    "ingredient_tag_review_samples": review_samples,
    "mapping_disagreements_low_confidence": mapping_disagreements[:25],
}

out = ROOT / "scratch" / "tag" / "_fp_analysis_output.json"
out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(json.dumps(report, indent=2, default=str))

