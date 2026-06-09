"""LLM fallback for ambiguous ingredient quantity/unit parsing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from parse_recipe_ingredient import PARSE_FIELDS, empty_parse_result
from openai_fallback import get_async_openai_client
from recipe_parse_rules import ALLOWED_UNITS

PROMPT_VERSION = "parse_v1"

MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You parse one recipe ingredient line into structured fields for nutrition lookup.\n"
    "Output JSON only. Use US customary units when applicable.\n"
    "Rules:\n"
    "- Set measurable=false for vague amounts (e.g. 'salt to taste', 'pinch' with no number).\n"
    "- unit must be one of the allowed units or null.\n"
    "- name is the food identity without quantity/unit/size/prep when possible.\n"
    "- certainty is 0.0-1.0 confidence in your parse."
)

RESPONSE_SCHEMA = {
    "name": "ingredient_parse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "quantity": {"type": ["number", "null"]},
            "quantity_max": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "name": {"type": "string"},
            "size": {"type": ["string", "null"]},
            "preparation": {"type": ["string", "null"]},
            "measurable": {"type": "boolean"},
            "certainty": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": [
            "quantity",
            "quantity_max",
            "unit",
            "name",
            "size",
            "preparation",
            "measurable",
            "certainty",
            "rationale",
        ],
        "additionalProperties": False,
    },
}

_ALLOWED_UNIT_SET = set(ALLOWED_UNITS)


def build_user_prompt(ingredient: str) -> str:
    units = ", ".join(ALLOWED_UNITS)
    return (
        f"INGREDIENT: {ingredient}\n\n"
        f"ALLOWED_UNITS: {units}\n\n"
        "Parse this ingredient line."
    )


def validate_llm_parse(parsed: dict[str, Any]) -> str | None:
    """Return error string if invalid, else None."""
    unit = parsed.get("unit")
    if unit is not None:
        unit_norm = str(unit).strip().lower()
        if unit_norm not in _ALLOWED_UNIT_SET:
            return f"invalid_unit:{unit_norm}"

    certainty = parsed.get("certainty")
    if certainty is not None:
        try:
            c = float(certainty)
            if c < 0 or c > 1:
                return "certainty_out_of_range"
        except (TypeError, ValueError):
            return "invalid_certainty"

    name = parsed.get("name")
    if not name or not str(name).strip():
        return "missing_name"

    return None


def llm_parse_to_fields(
    parsed: dict[str, Any],
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """Map validated LLM JSON to unified parse fields."""
    result = empty_parse_result()
    result["parse_method"] = "llm"

    if error:
        result["parse_status"] = "llm_invalid" if error != "api_error" else "error"
        result["name"] = parsed.get("name")
        result["confidence"] = parsed.get("certainty")
        result["measurable"] = parsed.get("measurable")
        return result

    measurable = bool(parsed.get("measurable"))
    qty = parsed.get("quantity")
    name = str(parsed.get("name", "")).strip() or None

    result.update({
        "quantity": float(qty) if qty is not None else None,
        "quantity_max": (
            float(parsed["quantity_max"])
            if parsed.get("quantity_max") is not None
            else (float(qty) if qty is not None else None)
        ),
        "unit": parsed.get("unit"),
        "unit_raw": parsed.get("unit"),
        "amount_text": None,
        "name": name,
        "size": parsed.get("size"),
        "preparation": parsed.get("preparation"),
        "confidence": parsed.get("certainty"),
        "measurable": measurable,
    })

    if not measurable:
        result["parse_status"] = "unmeasurable"
    elif name and qty is not None:
        result["parse_status"] = "ok"
    elif name:
        result["parse_status"] = "no_amount"
    else:
        result["parse_status"] = "ambiguous"

    return result


async def parse_one_async(
    client: Any,
    model: str,
    ingredient: str,
) -> dict[str, Any]:
    """One LLM parse call with validation retry."""

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

        validation_error = validate_llm_parse(parsed)
        if validation_error:
            hint = (
                f"Validation failed ({validation_error}). "
                "Use only allowed units or null. Fix and resubmit."
            )
            parsed, raw_response, usage = await _invoke(hint)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            validation_error = validate_llm_parse(parsed)
            if validation_error:
                error = validation_error
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    fields = llm_parse_to_fields(parsed, error=error)
    cost_input = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])["input"]
    cost_output = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])["output"]
    cost = prompt_tokens / 1e6 * cost_input + completion_tokens / 1e6 * cost_output

    return {
        "ingredient_norm": normalize_ingredient_key(ingredient),
        "ingredient_raw": ingredient,
        "fields": fields,
        "llm_certainty": parsed.get("certainty"),
        "llm_rationale": parsed.get("rationale"),
        "llm_measurable": parsed.get("measurable"),
        "llm_error": error,
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "price_estimate_usd": round(cost, 8),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(ingredient),
    }


def normalize_ingredient_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


async def run_llm_parses(
    ingredients: list[str],
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Parse unique ingredients concurrently; return cache + call log rows."""
    unique = list(dict.fromkeys(normalize_ingredient_key(t) for t in ingredients if str(t).strip()))
    key_to_raw: dict[str, str] = {}
    for text in ingredients:
        key = normalize_ingredient_key(text)
        if key and key not in key_to_raw:
            key_to_raw[key] = str(text).strip()

    client = get_async_openai_client()
    sem = asyncio.Semaphore(concurrency)
    cache: dict[str, dict[str, Any]] = {}
    call_rows: list[dict[str, Any]] = []

    async def _one(key: str) -> None:
        async with sem:
            raw = key_to_raw.get(key, key)
            row = await parse_one_async(client, model, raw)
            cache[key] = row
            call_rows.append(row)

    await asyncio.gather(*[_one(k) for k in unique])
    return cache, call_rows


def run_llm_parses_sync(
    ingredients: list[str],
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Synchronous wrapper for run_llm_parses."""
    return asyncio.run(
        run_llm_parses(ingredients, model=model, concurrency=concurrency)
    )


def pricing_for_model(model: str) -> dict[str, float]:
    return MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
