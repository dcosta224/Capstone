"""Parse RecipeNLG ingredient lines into quantity, unit, and name.

Approach A (library-only): uses `ingredient-parser-nlp` for all rows.
A rule-based fallback (Approach B) can be added later for rows with
parse_status in ('error', 'ambiguous') if needed.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

import pandas as pd
from ingredient_parser import parse_ingredient

PARSE_METHOD = "ingredient_parser"

PARSE_FIELDS = (
    "quantity",
    "quantity_max",
    "unit",
    "unit_raw",
    "amount_text",
    "name",
    "size",
    "preparation",
    "parse_status",
    "parse_method",
)

UNIFIED_RESULT_FIELDS = PARSE_FIELDS + (
    "confidence",
    "measurable",
    "final_method",
)

# Explicit vague / non-numeric recipe amounts (measurable=false without LLM).
_UNMEASURABLE_RE = re.compile(
    r"\b("
    r"to taste|as needed|as desired|for garnish|for serving|"
    r"if desired|or more|or less"
    r")\b",
    re.IGNORECASE,
)

_LEADING_QTY_RE = re.compile(
    r"^\s*("
    r"(?:\d+\s+)?\d+/\d+|"
    r"\d+(?:\.\d+)?|"
    r"\d+/\d+"
    r")",
)

# Leading numeric quantity (mixed number, fraction, or decimal) plus trailing space.
_QTY_PREFIX_RE = re.compile(
    r"(?:"
    r"(?:\d+\s+)?\d+/\d+|"  # 1 1/2 or 1/2
    r"\d+(?:\.\d+)?|"  # 2 or 2.5
    r"\d+/\d+"
    r")\s*"
)


def _to_float(value: Fraction | float | int | str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_amounts(amounts: list[Any]) -> list[Any]:
    """Expand composite amounts (e.g. '2 c plus 2 Tbsp.') into leaf amounts."""
    flat: list[Any] = []
    for amount in amounts:
        nested = getattr(amount, "amounts", None)
        if nested:
            flat.extend(_flatten_amounts(nested))
        else:
            flat.append(amount)
    return flat


def _primary_amount(amounts: list[Any]) -> Any | None:
    if not amounts:
        return None
    flat = _flatten_amounts(amounts)
    for amount in flat:
        if getattr(amount, "unit", None):
            return amount
    return flat[0] if flat else amounts[0]


def _join_name(names: list[Any]) -> str | None:
    texts = [n.text.strip() for n in names if n.text and n.text.strip()]
    return ", ".join(texts) if texts else None


def parse_ingredient_fields(text: str) -> dict[str, Any]:
    """Parse one ingredient string into structured fields."""
    empty = {field: None for field in PARSE_FIELDS}
    empty["parse_status"] = "empty"
    empty["parse_method"] = None

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return empty
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return empty

    try:
        parsed = parse_ingredient(
            text,
            string_units=True,
            volumetric_units_system="us_customary",
        )
    except Exception:
        result = empty.copy()
        result["name"] = text
        result["parse_status"] = "error"
        result["parse_method"] = PARSE_METHOD
        return result

    amount = _primary_amount(parsed.amount)
    name = _join_name(parsed.name)

    result: dict[str, Any] = {
        "quantity": _to_float(amount.quantity) if amount else None,
        "quantity_max": _to_float(amount.quantity_max) if amount else None,
        "unit": (amount.unit or None) if amount else None,
        "unit_raw": amount.text if amount else None,
        "amount_text": ", ".join(a.text for a in parsed.amount) if parsed.amount else None,
        "name": name,
        "size": parsed.size.text if parsed.size else None,
        "preparation": parsed.preparation.text if parsed.preparation else None,
        "parse_status": None,
        "parse_method": PARSE_METHOD,
    }

    if name and amount:
        result["parse_status"] = "ok"
    elif name:
        result["parse_status"] = "no_amount"
    else:
        result["parse_status"] = "ambiguous"
        result["name"] = text

    return result


def empty_parse_result() -> dict[str, Any]:
    result = {field: None for field in PARSE_FIELDS}
    result["parse_status"] = "empty"
    result["parse_method"] = None
    result["confidence"] = None
    result["measurable"] = None
    result["final_method"] = None
    return result


def is_explicitly_unmeasurable(text: str) -> bool:
    """True when raw text contains vague amount phrases (to taste, or more, etc.)."""
    if not text or not str(text).strip():
        return False
    return bool(_UNMEASURABLE_RE.search(str(text)))


def has_ok_parse(result: dict[str, Any]) -> bool:
    """True when the parser produced a usable structured amount."""
    status = result.get("parse_status")
    return status in ("ok", "parsed_size_as_count") and has_usable_amount(result)


def should_force_unmeasurable(result: dict[str, Any], *, ingredient_raw: str = "") -> bool:
    """Downgrade to unmeasurable only when vague phrasing and parse did not succeed."""
    if not ingredient_raw or not is_explicitly_unmeasurable(ingredient_raw):
        return False
    return not has_ok_parse(result)


def is_measurable(result: dict[str, Any], *, ingredient_raw: str = "") -> bool:
    """False when the line is explicitly vague or marked unmeasurable."""
    if result.get("measurable") is False:
        return False
    if ingredient_raw and should_force_unmeasurable(result, ingredient_raw=ingredient_raw):
        return False
    return True


def has_usable_amount(result: dict[str, Any]) -> bool:
    return result.get("quantity") is not None and bool(result.get("name"))


def needs_llm_fallback(result: dict[str, Any], *, ingredient_raw: str = "") -> bool:
    """Trigger LLM when no usable measurable amount was extracted."""
    if should_force_unmeasurable(result, ingredient_raw=ingredient_raw):
        return False
    status = result.get("parse_status")
    if status in ("error", "ambiguous", "no_amount", "empty", "no_quantity"):
        return True
    if status in ("ok", "parsed_size_as_count") and has_usable_amount(result):
        return False
    if result.get("name") and result.get("quantity") is None:
        return True
    return status in ("quantity_no_known_unit", "llm_invalid")


def merge_parse_result(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    final_method: str,
) -> dict[str, Any]:
    """Merge override into base; override wins on non-null fields."""
    merged = empty_parse_result()
    for field in PARSE_FIELDS:
        val = override.get(field)
        if val is None:
            val = base.get(field)
        merged[field] = val
    merged["confidence"] = override.get("confidence", base.get("confidence"))
    measurable = override.get("measurable")
    if measurable is None:
        measurable = base.get("measurable")
    merged["measurable"] = measurable
    merged["final_method"] = final_method
    if merged.get("parse_status") in (None, "empty"):
        merged["parse_status"] = override.get("parse_status") or base.get("parse_status")
    return merged


def finalize_parse_result(
    result: dict[str, Any],
    *,
    ingredient_raw: str,
    final_method: str,
) -> dict[str, Any]:
    """Attach measurable/final_method and normalize terminal status."""
    out = dict(result)
    out["final_method"] = final_method
    if should_force_unmeasurable(out, ingredient_raw=ingredient_raw):
        out["measurable"] = False
    elif out.get("measurable") is None:
        out["measurable"] = has_usable_amount(out)
    if out.get("parse_status") in (None, "empty") and has_usable_amount(out):
        out["parse_status"] = "ok"
    return out


def looks_parseable_raw(text: str) -> bool:
    """Heuristic: line starts with a numeric quantity."""
    return bool(_LEADING_QTY_RE.match(str(text or "")))


def _clean_dequant(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip(" ,")


def _join_parsed_name(parsed: Any) -> str:
    names = parsed.name or []
    texts = [n.text.strip() for n in names if n.text and n.text.strip()]
    return ", ".join(texts)


def _ingredient_name_start(text: str, parsed: Any) -> int:
    """Index where the ingredient name begins (amount region is text[:start])."""
    name_text = _join_parsed_name(parsed)
    if name_text:
        idx = text.casefold().find(name_text.casefold())
        return idx if idx >= 0 else len(text)
    if parsed.size and parsed.size.text:
        idx = text.casefold().find(parsed.size.text.casefold())
        if idx > 0:
            return idx
    return len(text)


def _is_valid_quantity_match(text: str, match: re.Match[str]) -> bool:
    """Reject matches that are a size suffix (e.g. the 2 in ``2-inch``), not a recipe quantity."""
    matched = match.group()
    if matched != matched.rstrip():
        return True  # quantity followed by whitespace — delimited from next token
    end = match.end()
    if end >= len(text):
        return True
    if text[end] == "-":
        return False
    return True


def _qty_span_at_index(text: str, start: int) -> tuple[int, int] | None:
    if start < 0 or start >= len(text):
        return None
    match = _QTY_PREFIX_RE.match(text, start)
    if not match or not _is_valid_quantity_match(text, match):
        return None
    return match.start(), match.end()


def _amounts_to_strip(amounts: list[Any]) -> list[Any]:
    """
    Amount objects whose numeric quantity should be removed from the raw text.

    - One composite amount (``2 c plus 2 Tbsp``): every nested leaf quantity.
    - Multiple top-level amounts (``4 4-oz chicken``): only the leading recipe count.
    - Otherwise: the single top-level amount.
    """
    if not amounts:
        return []

    if len(amounts) == 1:
        one = amounts[0]
        if getattr(one, "amounts", None):
            return [
                a for a in _flatten_amounts([one])
                if getattr(a, "quantity", None) is not None
            ]
        return [one] if getattr(one, "quantity", None) is not None else []

    first = amounts[0]
    if getattr(first, "amounts", None):
        flat = [
            a for a in _flatten_amounts([first])
            if getattr(a, "quantity", None) is not None
        ]
        return flat[:1] if flat else []
    return [first] if getattr(first, "quantity", None) is not None else []


def _spans_in_range(text: str, start: int, end: int, n: int) -> list[tuple[int, int]]:
    """Up to ``n`` quantity spans found left-to-right within ``text[start:end)``."""
    spans: list[tuple[int, int]] = []
    pos = max(0, start)
    end = min(len(text), end)
    while len(spans) < n and pos < end:
        match = _QTY_PREFIX_RE.match(text, pos)
        if match and _is_valid_quantity_match(text, match):
            spans.append((match.start(), match.end()))
            pos = match.end()
        else:
            pos += 1
    return spans


def _quantity_spans(text: str, parsed: Any) -> list[tuple[int, int]]:
    """Character spans of recipe quantities to remove (parser-guided, not all digits)."""
    amounts = parsed.amount or []
    to_strip = _amounts_to_strip(amounts)
    if not to_strip:
        return []

    # Composite recipe amount: strip each leaf quantity before the ingredient name.
    if len(amounts) == 1 and getattr(amounts[0], "amounts", None):
        parent = amounts[0]
        start = parent.starting_index or 0
        end = _ingredient_name_start(text, parsed)
        return _spans_in_range(text, start, end, len(to_strip))

    spans: list[tuple[int, int]] = []
    for amount in to_strip:
        start = getattr(amount, "starting_index", None)
        if start is None:
            continue
        span = _qty_span_at_index(text, start)
        if span:
            spans.append(span)
    return spans


def strip_quantities_from_text(text: str) -> str:
    """
    Return original ingredient text with parsed numeric quantities removed.

    Units, size, preparation, and wording are preserved; only quantity numerals
    identified by ``ingredient-parser-nlp`` are stripped.
    """
    text = (text or "").strip()
    if not text:
        return ""

    try:
        parsed = parse_ingredient(
            text,
            string_units=True,
            volumetric_units_system="us_customary",
        )
    except Exception:
        return text

    if not parsed.amount:
        return text

    spans = _quantity_spans(text, parsed)
    if not spans:
        return text

    out = text
    for start, end in reversed(spans):
        out = out[:start] + out[end:]
    return _clean_dequant(out)
