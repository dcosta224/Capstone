"""LLM fallback to pick a USDA food_portion row when rules-based resolve_grams fails."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ingredient_parse_llm import DEFAULT_MODEL, MODEL_PRICING
from llm_throttle import throttle_llm_async, throttle_llm_sync
from portion_gram import (
    CountPortionCandidate,
    PortionCandidate,
    PortionGramResult,
    _load_portion_rows_for_fdc,
    _portion_text_fields,
    classify_food_portion_row,
    normalize_count_portion_row,
    normalize_portion_row,
    pick_best_count_portion,
    pick_best_portion,
    resolve_grams,
)
from amount_kind import infer_count_query
from unit_convert import UnitConversionError, convert_volume, normalize_volume_unit

PROMPT_VERSION = "portion_pick_v2"

SYSTEM_PROMPT = (
    "Pick the best USDA food_portion row to convert a recipe ingredient amount to grams.\n"
    "Each portion line shows measure_unit, modifier, portion_description, amount, gram_weight, "
    "and a rules_class tag (volume/count/mass/other). Include mass rows with container modifiers "
    "(e.g. can/jar) when recipe unit is a container.\n"
    "Good-enough matching is OK: exact unit/size match preferred, but acceptable fallbacks:\n"
    "- Recipe names size (small/medium/large) -> pick that modifier.\n"
    "- Generic count (3 eggs, 10 mushrooms) -> medium, then large, then small.\n"
    "- Container unit (can) -> modifier containing can even if rules_class=mass.\n"
    "If none are close enough, set portion_id to null.\n"
    "Keep rationale to one short sentence. certainty is 0.0-1.0."
)

RESPONSE_SCHEMA = {
    "name": "portion_pick",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "portion_id": {"type": ["integer", "null"]},
            "certainty": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["portion_id", "certainty", "rationale"],
        "additionalProperties": False,
    },
}


def format_portion_options(rows: list[dict[str, Any]], *, max_rows: int = 12) -> str:
    lines: list[str] = []
    for row in rows[:max_rows]:
        modifier, desc, mu = _portion_text_fields(row)
        tag = classify_food_portion_row(row)
        lines.append(
            f"id={row['id']} | unit={mu or '-'} | modifier={modifier or '-'} | "
            f"desc={desc or '-'} | amount={row.get('amount')} | "
            f"gram_weight={row.get('gram_weight')} | rules_class={tag}"
        )
    return "\n".join(lines) if lines else "(no portion rows)"


def build_user_prompt(
    *,
    ingredient: str,
    quantity: float,
    unit: str | None,
    name: str | None,
    amount_kind: str,
    fdc_id: int,
    portion_block: str,
) -> str:
    return (
        f"INGREDIENT: {ingredient}\n"
        f"PARSED: quantity={quantity}; unit={unit or '-'}; name={name or '-'}; "
        f"amount_kind={amount_kind}\n"
        f"FDC_ID: {fdc_id}\n\n"
        f"PORTION OPTIONS:\n{portion_block}\n\n"
        "Select portion_id."
    )


def _grams_from_volume_portion(
    q: float, unit_str: str, portion: PortionCandidate
) -> float | None:
    recipe_ml = convert_volume(q, unit_str, "milliliter")
    ref_ml = convert_volume(portion.ref_amount, portion.ref_unit, "milliliter")
    if ref_ml <= 0:
        return None
    return round((recipe_ml / ref_ml) * portion.gram_weight, 4)


def _grams_from_count_portion(
    q: float, portion: CountPortionCandidate
) -> float | None:
    if portion.ref_amount <= 0:
        return None
    return round((q / portion.ref_amount) * portion.gram_weight, 4)


def apply_portion_pick(
    portion_id: int,
    raw_rows: list[dict[str, Any]],
    *,
    quantity: float,
    unit: str | None,
    name: str | None,
    amount_kind: str,
) -> PortionGramResult:
    row = next((r for r in raw_rows if int(r["id"]) == int(portion_id)), None)
    if row is None:
        return PortionGramResult(
            grams=None,
            status="no_portion",
            unit_kind=amount_kind,
            method="llm picked missing portion_id",
        )

    q = float(quantity)
    unit_str = str(unit) if unit is not None else ""

    vol = normalize_portion_row(row)
    if vol is not None and amount_kind == "volume":
        try:
            normalize_volume_unit(unit_str or vol.ref_unit)
            grams = _grams_from_volume_portion(q, unit_str or vol.ref_unit, vol)
            if grams is not None:
                return PortionGramResult(
                    grams=grams,
                    status="ok_volume_portion_llm",
                    unit_kind="volume",
                    portion_id=vol.portion_id,
                    portion_ref_amount=vol.ref_amount,
                    portion_ref_unit=vol.ref_unit,
                    method=f"llm_volume:portion#{vol.portion_id}",
                )
        except UnitConversionError:
            pass

    cnt = normalize_count_portion_row(row)
    if cnt is not None and amount_kind == "count":
        grams = _grams_from_count_portion(q, cnt)
        if grams is not None:
            return PortionGramResult(
                grams=grams,
                status="ok_count_portion_llm",
                unit_kind="count",
                portion_id=cnt.portion_id,
                portion_ref_amount=cnt.ref_amount,
                portion_ref_unit=cnt.count_label,
                method=f"llm_count:portion#{cnt.portion_id}",
            )

    return PortionGramResult(
        grams=None,
        status="no_portion",
        unit_kind=amount_kind,
        portion_id=int(portion_id),
        method="llm pick did not convert",
    )


def pick_portion_sync(
    model: str,
    *,
    ingredient: str,
    quantity: float,
    unit: str | None,
    name: str | None,
    amount_kind: str,
    fdc_id: int,
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sync OpenAI client — safe inside asyncio judging workers."""
    from openai_fallback import get_sync_openai_client

    client = get_sync_openai_client()
    block = format_portion_options(raw_rows)
    user = build_user_prompt(
        ingredient=ingredient,
        quantity=quantity,
        unit=unit,
        name=name,
        amount_kind=amount_kind,
        fdc_id=fdc_id,
        portion_block=block,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content)
    usage = resp.usage
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    cost = usage.prompt_tokens / 1e6 * pricing["input"] + usage.completion_tokens / 1e6 * pricing["output"]
    return {
        "portion_id": parsed.get("portion_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "price_estimate_usd": round(cost, 8),
        "user_prompt": user,
    }


async def pick_portion_async(
    client: Any,
    model: str,
    *,
    ingredient: str,
    quantity: float,
    unit: str | None,
    name: str | None,
    amount_kind: str,
    fdc_id: int,
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    block = format_portion_options(raw_rows)
    user = build_user_prompt(
        ingredient=ingredient,
        quantity=quantity,
        unit=unit,
        name=name,
        amount_kind=amount_kind,
        fdc_id=fdc_id,
        portion_block=block,
    )

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content)
    usage = resp.usage
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    cost = usage.prompt_tokens / 1e6 * pricing["input"] + usage.completion_tokens / 1e6 * pricing["output"]
    return {
        "portion_id": parsed.get("portion_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "price_estimate_usd": round(cost, 8),
        "user_prompt": user,
    }


def resolve_grams_with_llm_fallback(
    fdc_id: int | None,
    quantity: float | None,
    unit: str | None,
    *,
    name: str | None = None,
    ingredient_raw: str | None = None,
    amount_kind: str | None = None,
    portion_index: dict | None = None,
    count_portion_index: dict | None = None,
    conn=None,
    llm_picker: Any | None = None,
) -> tuple[PortionGramResult, dict[str, Any] | None]:
    """Rules-based resolve_grams; optional async LLM picker on no_portion for volume/count."""
    result = resolve_grams(
        fdc_id,
        quantity,
        unit,
        name=name,
        ingredient_raw=ingredient_raw,
        amount_kind=amount_kind,
        portion_index=portion_index,
        count_portion_index=count_portion_index,
        conn=conn,
    )
    meta: dict[str, Any] | None = None
    if result.status != "no_portion" or amount_kind not in ("volume", "count"):
        return result, meta
    if fdc_id is None or quantity is None or llm_picker is None or conn is None:
        return result, meta

    raw_rows = _load_portion_rows_for_fdc(conn, int(fdc_id))
    if not raw_rows:
        return result, meta

    llm_meta = llm_picker(
        ingredient=str(ingredient_raw or ""),
        quantity=float(quantity),
        unit=unit,
        name=name,
        amount_kind=str(amount_kind),
        fdc_id=int(fdc_id),
        raw_rows=raw_rows,
    )
    meta = llm_meta
    pid = llm_meta.get("portion_id")
    if pid is None:
        result = PortionGramResult(
            grams=None,
            status="no_portion",
            unit_kind=amount_kind,
            method="llm abstained on portion pick",
        )
        return result, meta

    llm_result = apply_portion_pick(
        int(pid),
        raw_rows,
        quantity=float(quantity),
        unit=unit,
        name=name,
        amount_kind=str(amount_kind),
    )
    return llm_result, meta
