"""LLM fallback to classify ingredient amount kind when rules return unknown."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from amount_kind import AmountKind, classify_amount_kind
from ingredient_parse_llm import DEFAULT_MODEL, MODEL_PRICING, normalize_ingredient_key
from openai_fallback import get_async_openai_client

PROMPT_VERSION = "amount_kind_v1"

SYSTEM_PROMPT = (
    "Classify one recipe ingredient line by how its amount should be resolved to grams.\n"
    "amount_kind must be one of:\n"
    "- mass: explicit mass unit (g, oz, lb, kg)\n"
    "- volume: explicit volume unit (cup, tsp, tbsp, ml, L, pint, quart)\n"
    "- count: discrete count (2 eggs, 1 slice, 3 cloves) with or without a count unit\n"
    "- unmeasurable: no usable amount (to taste, pinch without number)\n"
    "- unknown: truly ambiguous\n"
    "Keep rationale to one short sentence. certainty is 0.0-1.0."
)

RESPONSE_SCHEMA = {
    "name": "amount_kind_classify",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "amount_kind": {
                "type": "string",
                "enum": ["mass", "volume", "count", "unmeasurable", "unknown"],
            },
            "quantity": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "name": {"type": "string"},
            "certainty": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["amount_kind", "quantity", "unit", "name", "certainty", "rationale"],
        "additionalProperties": False,
    },
}


def build_user_prompt(ingredient: str) -> str:
    return f"INGREDIENT: {ingredient}\n\nClassify amount_kind and parse quantity/unit/name if present."


def validate_response(parsed: dict[str, Any]) -> str | None:
    kind = parsed.get("amount_kind")
    if kind not in ("mass", "volume", "count", "unmeasurable", "unknown"):
        return "invalid_amount_kind"
    try:
        c = float(parsed["certainty"])
        if c < 0 or c > 1:
            return "certainty_out_of_range"
    except (TypeError, ValueError, KeyError):
        return "invalid_certainty"
    if not str(parsed.get("name", "")).strip():
        return "missing_name"
    return None


async def classify_one_async(client: Any, model: str, ingredient: str) -> dict[str, Any]:
    async def _invoke(extra: str | None = None):
        user = build_user_prompt(ingredient)
        if extra:
            user += f"\n\n{extra}"
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            temperature=0,
        )
        content = resp.choices[0].message.content
        return json.loads(content), content, resp.usage

    prompt_tokens = completion_tokens = 0
    error: str | None = None
    parsed: dict[str, Any] = {}
    raw_response: str | None = None

    try:
        parsed, raw_response, usage = await _invoke()
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens
        validation_error = validate_response(parsed)
        if validation_error:
            parsed, raw_response, usage = await _invoke(
                f"Validation failed ({validation_error}). Fix and resubmit."
            )
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            validation_error = validate_response(parsed)
            if validation_error:
                error = validation_error
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    cost = prompt_tokens / 1e6 * pricing["input"] + completion_tokens / 1e6 * pricing["output"]

    return {
        "ingredient_norm": normalize_ingredient_key(ingredient),
        "ingredient_raw": ingredient,
        "amount_kind": parsed.get("amount_kind"),
        "quantity": parsed.get("quantity"),
        "unit": parsed.get("unit"),
        "name": parsed.get("name"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "error": error,
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "price_estimate_usd": round(cost, 8),
    }


def run_amount_kind_llm_sync(
    ingredients: list[str],
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    unique = list(dict.fromkeys(normalize_ingredient_key(t) for t in ingredients if str(t).strip()))
    norm_to_raw: dict[str, str] = {}
    for t in ingredients:
        k = normalize_ingredient_key(t)
        if k and k not in norm_to_raw:
            norm_to_raw[k] = t

    async def _run():
        client = get_async_openai_client()
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []

        async def _one(norm: str):
            async with sem:
                return await classify_one_async(client, model, norm_to_raw[norm])

        tasks = [_one(n) for n in unique]
        return await asyncio.gather(*tasks)

    rows = asyncio.run(_run())
    cache = {r["ingredient_norm"]: r for r in rows}
    return cache, rows


def classify_amount_kind_with_llm_fallback(
    ingredient: str,
    rules_fields: dict[str, Any],
    *,
    llm_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[AmountKind, str, dict[str, Any]]:
    """Rules first; LLM cache only when rules yield unknown."""
    kind: AmountKind = classify_amount_kind(
        rules_fields.get("quantity"),
        rules_fields.get("unit"),
        rules_fields.get("name"),
        ingredient_raw=ingredient,
        parse_status=rules_fields.get("parse_status"),
    )
    meta: dict[str, Any] = {
        "amount_kind_source": "rules",
        "amount_kind_llm_certainty": None,
        "amount_kind_llm_rationale": None,
    }
    if kind != "unknown" or not llm_cache:
        return kind, "rules", meta

    key = normalize_ingredient_key(ingredient)
    llm_row = llm_cache.get(key)
    if not llm_row or llm_row.get("error") or not llm_row.get("amount_kind"):
        return kind, "rules", meta

    llm_kind = llm_row["amount_kind"]
    meta.update({
        "amount_kind_source": "llm",
        "amount_kind_llm_certainty": llm_row.get("certainty"),
        "amount_kind_llm_rationale": llm_row.get("rationale"),
        "llm_quantity": llm_row.get("quantity"),
        "llm_unit": llm_row.get("unit"),
        "llm_name": llm_row.get("name"),
    })
    return llm_kind, "llm", meta
