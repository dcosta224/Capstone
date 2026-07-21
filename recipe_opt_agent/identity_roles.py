"""Identity role extraction: templates ∪ lexical/LLM merge."""

from __future__ import annotations

import os
from typing import Any

from recipe_opt_agent.config import IDENTITY_TEMPLATES, identity_roles_for_title

# Lightweight dish → roles for free-text titles not in IDENTITY_TEMPLATES.
_LEXICAL_DISH_ROLES: dict[str, list[str]] = {
    "shakshuka": ["egg", "tomato", "pepper", "onion"],
    "ramen": ["noodle", "broth", "protein"],
    "risotto": ["rice", "stock", "cheese"],
    "pho": ["noodle", "broth", "herb"],
    "pad thai": ["noodle", "protein", "tamarind"],
    "burrito": ["tortilla", "bean", "protein"],
    "taco": ["tortilla", "protein"],
    "curry": ["sauce", "protein", "aromatic"],
    "chili": ["bean", "protein", "chili"],
    "salad": ["greens", "dressing"],
    "soup": ["broth", "vegetable"],
    "stew": ["protein", "vegetable", "broth"],
    "omelette": ["egg"],
    "omelet": ["egg"],
    "pancake": ["batter", "egg"],
    "stir fry": ["vegetable", "protein", "sauce"],
    "stir-fry": ["vegetable", "protein", "sauce"],
}


def lexical_roles_from_text(title: str, request: str = "") -> list[str]:
    text = f"{title} {request}".lower()
    for key, roles in _LEXICAL_DISH_ROLES.items():
        if key in text:
            return list(roles)
    # Ingredient-ish cues
    roles: list[str] = []
    cues = [
        ("egg", "egg"),
        ("tomato", "tomato"),
        ("cheese", "cheese"),
        ("pasta", "pasta"),
        ("noodle", "noodle"),
        ("rice", "rice"),
        ("chicken", "protein"),
        ("beef", "protein"),
        ("tofu", "protein"),
        ("pork", "cured_pork"),
        ("bacon", "cured_pork"),
        ("crust", "crust"),
        ("dough", "crust"),
        ("bean", "bean"),
        ("lentil", "bean"),
    ]
    for needle, role in cues:
        if needle in text and role not in roles:
            roles.append(role)
    return roles


def merge_roles(*role_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in role_lists:
        for r in lst or []:
            key = str(r).strip().lower().replace(" ", "_")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def resolve_identity_roles(
    *,
    title: str,
    request: str = "",
    ingredients: list[dict[str, Any]] | None = None,
    templates: dict[str, list[str]] | None = None,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Template ∪ lexical ∪ optional LLM roles."""
    template = identity_roles_for_title(title or request, templates or IDENTITY_TEMPLATES)
    lexical = lexical_roles_from_text(title or "", request or "")
    if ingredients:
        labels = " ".join(str(r.get("label") or r.get("name") or "") for r in ingredients)
        lexical = merge_roles(lexical, lexical_roles_from_text(labels, ""))
    extracted: list[str] = []
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            extracted = extract_identity_roles_llm(
                title=title,
                request=request,
                ingredients=ingredients,
                model=model,
            )
        except Exception:
            extracted = []
    return merge_roles(template, lexical, extracted)


def extract_identity_roles_llm(
    *,
    title: str,
    request: str = "",
    ingredients: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """LLM JSON list of stable role ids (pasta, egg, cheese, …)."""
    from recipe_opt_agent.llm import _call_json_llm

    ings = ingredients or []
    labels = [str(r.get("label") or r.get("name") or "") for r in ings][:20]
    system = (
        "Extract dish-identity ingredient ROLES that must remain filled for this dish to stay itself. "
        "Return JSON: {\"identity_roles\": [\"role\", ...]}. "
        "Use short stable ids like pasta, egg, cheese, cured_pork, crust, tomato, protein, broth, noodle. "
        "Prefer 2–6 roles. Do not invent roles unrelated to the dish."
    )
    user = (
        f"Title: {title}\nRequest: {request}\nIngredients: {labels}\n"
        "Respond with identity_roles JSON only."
    )

    def _heuristic() -> dict[str, Any]:
        return {"identity_roles": lexical_roles_from_text(title, request)}

    data, _ = _call_json_llm(system=system, user=user, model=model, heuristic_fn=_heuristic)
    roles = data.get("identity_roles") or []
    return [str(r).strip().lower().replace(" ", "_") for r in roles if str(r).strip()]
