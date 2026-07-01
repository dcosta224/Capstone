import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from diet_tags_core import detect_contains, load_diet_tags

df = pd.read_csv(ROOT / "scratch/tag/foodon_contains.csv")
df["label_lc"] = df["label"].astype(str).str.lower()
SLUGS = [
    "alcohol", "dairy", "egg", "fish", "honey", "peanut", "pork", "poultry",
    "red_meat", "root_vegetable", "sesame", "shellfish", "soy", "tree_nut", "wheat",
]

reg = load_diet_tags()

def pick(fps, slug, mask, why, limit=4):
    sub = df[mask]
    for _, r in sub.iterrows():
        if len(fps[slug]) >= limit:
            return
        fid = r["foodon_id"]
        if any(x["foodon_id"] == fid for x in fps[slug]):
            continue
        fps[slug].append({
            "foodon_id": fid,
            "label": r["label"],
            "wrong_tag": slug,
            "why": why,
            "source_type": "FoodOn propagation",
        })

fps = {s: [] for s in SLUGS}

pick(fps, "dairy", df["contains_dairy"] & df["label_lc"].str.contains(
    r"coconut milk|almond milk|oat milk|rice milk|soy milk|plant-based|dairy-free|non-dairy", regex=True, na=False),
    "Plant or dairy-free milk/dessert node inherits contains_dairy from dairy food product subtree")

pick(fps, "dairy", df["contains_dairy"] & df["label_lc"].str.contains(
    r"butter bean|butternut|cocoa butter|shea butter|peanut butter", regex=True, na=False),
    "Non-dairy \"butter\" in FoodOn label still under dairy/milk product hierarchy")

pick(fps, "tree_nut", df["contains_tree_nut"] & df["label_lc"].str.contains("coconut", na=False),
    "Coconut tagged tree_nut via nut food product ancestor (allergen taxonomy mismatch)")

pick(fps, "tree_nut", df["contains_tree_nut"] & df["label_lc"].str.contains(
    r"nutmeg|doughnut|nutrient", regex=True, na=False),
    "Label contains nut substring; inherits tree_nut from nut food product root")

pick(fps, "wheat", df["contains_wheat"] & df["label_lc"].str.contains("buckwheat", na=False),
    "Buckwheat is not wheat; tagged via wheat food product or grain grouping")

pick(fps, "wheat", df["contains_wheat"] & df["label_lc"].str.contains(
    r"^barley|^rye|^oat|^rice|^corn", regex=True, na=False),
    "Other grain labeled under wheat food product subtree")

pick(fps, "poultry", df["contains_poultry"] & df["label_lc"].str.contains(
    r"chicken (?:broth|stock|soup|flavor|extract)", regex=True, na=False),
    "Broth/flavor/soup nodes tagged poultry — may over-flag vegetarian items using chicken stock base")

pick(fps, "poultry", df["contains_poultry"] & ~df["label_lc"].str.contains(
    r"chicken|turkey|duck|goose|quail|poultry|fowl|hen|grouse|partridge|pheasant", regex=True, na=False),
    "contains_poultry=True but label lacks obvious bird-meat terms (ontology-only tag)")

pick(fps, "red_meat", df["contains_red_meat"] & df["label_lc"].str.contains(
    r"beef (?:broth|stock|extract|flavor)", regex=True, na=False),
    "Beef broth/extract/flavor inherits red_meat from cattle soup/meat ancestors")

pick(fps, "red_meat", df["contains_red_meat"] & df["label_lc"].str.contains("cow food product", na=False),
    "Generic cow food product tagged red_meat (may include non-meat bovine products)")

pick(fps, "pork", df["contains_pork"] & df["label_lc"].str.contains("porterhouse", na=False),
    "porterhouse chop is beef cut but tagged pork — swine vs bovine ontology overlap false positive")

pick(fps, "pork", df["contains_pork"] & ~df["label_lc"].str.contains(
    r"pork|swine|ham|bacon|prosciutto|sausage|pig|pancetta|chorizo|salami", regex=True, na=False),
    "contains_pork without explicit swine terms (inherited from swine food product)")

pick(fps, "fish", df["contains_fish"] & df["label_lc"].str.contains(
    r"fish sauce|anchovy paste|bonito", regex=True, na=False),
    "Condiment/base fish products — tagged fish (often intended; note for FDC keyword overlap)")

pick(fps, "fish", df["contains_fish"] & df["label_lc"].str.contains("goldfish", na=False),
    "Non-food fish name would inherit fish tag if present in ontology")

pick(fps, "shellfish", df["contains_shellfish"] & df["label_lc"].str.contains(
    r"krill|seafood mix|surimi", regex=True, na=False),
    "Shellfish subtree includes processed seafood blends")

pick(fps, "shellfish", df["contains_shellfish"] & ~df["label_lc"].str.contains(
    r"shrimp|crab|lobster|clam|mussel|oyster|scallop|crayfish|prawn|shellfish|crustacean", regex=True, na=False),
    "contains_shellfish without explicit shellfish terms in label")

pick(fps, "egg", df["contains_egg"] & df["label_lc"].str.contains("fish egg|caviar|roe", regex=True, na=False),
    "Fish roe tagged contains_egg via egg food product — poultry egg allergen mismatch")

pick(fps, "egg", df["contains_egg"] & df["label_lc"].str.contains("eggplant", na=False),
    "If eggplant node under egg product — keyword-style false positive risk")

pick(fps, "soy", df["contains_soy"] & df["label_lc"].str.contains("soy sauce", na=False),
    "Soy sauce tagged soy (correct ontology; FDC may double-hit via keywords)")

pick(fps, "soy", df["contains_soy"] & ~df["label_lc"].str.contains(
    r"soy|soya|tofu|edamame|miso|tempeh", regex=True, na=False),
    "contains_soy without soy terms in label")

pick(fps, "root_vegetable", df["contains_root_vegetable"] & df["label_lc"].str.contains(
    r"horseradish|ginger|turmeric|wasabi", regex=True, na=False),
    "Spice/aromatic roots tagged root_vegetable (dietary onion/garlic/potato axis)")

pick(fps, "honey", df["contains_honey"] & df["label_lc"].str.contains(
    r"honey mustard|honey garlic|honey bbq", regex=True, na=False),
    "Flavored products with honey in name — may be trace honey vs primary ingredient")

pick(fps, "alcohol", df["contains_alcohol"] & df["label_lc"].str.contains(
    r"extract|vinegar|non-alcoholic|alcohol-free|de-alcohol", regex=True, na=False),
    "Low/no-alcohol or cooking extract under alcoholic beverage subtree")

pick(fps, "sesame", df["contains_sesame"] & df["label_lc"].str.contains("sesame", na=False),
    "Sesame-tagged nodes (baseline); disagreements often from mapping not sesame itself")

pick(fps, "peanut", df["contains_peanut"] & df["label_lc"].str.contains("peanut butter", na=False),
    "True peanut lineage (not FP) — contrast with tree_nut/dairy keyword collisions on same FDC text")

# keyword samples
samples = [
    "black pepper", "bell pepper", "pepper", "peanut butter", "peanut butter sandwich",
    "butter", "hamburger", "chatham", "ham sandwich", "ice cream", "cream of mushroom soup",
    "sunscreen cream", "cream cheese", "doughnut", "donut glazed", "walnut", "eggplant parmesan",
    "red wine vinegar", "soy sauce", "chicken broth low sodium", "coconut milk beverage",
    "ROMANO GOURMET CUT CROUTONS, ROMANO", "LEMON PEPPER NATURAL CATCH GRILLED FILLETS, LEMON PEPPER",
    "ORGANIC CHOCOLATE HAZELNUT DECADENCE 100% PLANT BASED DAIRY-FREE FROZEN DESSERT",
]
kw = []
for desc in samples:
    hits = detect_contains(desc, None, reg)
    kw.append({"description": desc, "hits": hits})

# review csv
review_path = ROOT / "scratch/tag/ingredient_tag_review.csv"
rev = pd.read_csv(review_path)
rev["cache_only_adds"] = rev["cache_only_adds"].fillna("").astype(str)
rev["keyword_only_adds"] = rev["keyword_only_adds"].fillna("").astype(str)
major = ["dairy", "fish", "poultry", "wheat", "tree_nut", "pork", "shellfish", "root_vegetable", "egg", "soy"]
review_out = {}
for col in ["cache_only_adds", "keyword_only_adds"]:
    rows = []
    for slug in major:
        pat = rf"(^|,){re.escape(slug)}(,|$)"
        m = rev[rev[col].str.contains(pat, regex=True, na=False)]
        for _, r in m.head(10).iterrows():
            rows.append({
                "slug": slug,
                "fdc_id": int(r["fdc_id"]),
                "description": r["description"],
                "foodon_id": r.get("foodon_id"),
                "foodon_label": r.get("foodon_label"),
                "confidence": r.get("confidence"),
                "bucket": r.get("bucket"),
                col: r[col],
            })
    review_out[col] = rows

# mapping disagreements
rev["confidence"] = pd.to_numeric(rev["confidence"], errors="coerce")
dis = []
for _, r in rev[rev["confidence"] < 0.65].iterrows():
    kw_c = str(r.get("keyword_contains") or "")
    cache_c = str(r.get("cache_contains") or "")
    co = r.get("cache_only_adds")
    ko = r.get("keyword_only_adds")
    if kw_c != cache_c or (pd.notna(co) and str(co)) or (pd.notna(ko) and str(ko)):
        dis.append({
            "fdc_id": int(r["fdc_id"]),
            "description": r["description"],
            "foodon_id": r.get("foodon_id"),
            "foodon_label": r.get("foodon_label"),
            "confidence": float(r["confidence"]),
            "keyword_contains": kw_c,
            "cache_contains": cache_c,
            "cache_only_adds": co if pd.notna(co) else "",
            "keyword_only_adds": ko if pd.notna(ko) else "",
            "source_type": "bad partner mapping",
        })


