"""Optional local LLM judge for FoodOn class → dietary contains_* tagging."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Literal

from diet_tags_core import load_diet_tags

VALID_SLUGS = frozenset(load_diet_tags().contains.keys())

SYSTEM_PROMPT = (
    "You classify FoodOn ontology food classes for dietary restriction detection.\n"
    "For each item, choose zero or more contains dimensions that apply to the food itself "
    "(not processing equipment or abstract categories).\n"
    f"Valid dimensions: {', '.join(sorted(VALID_SLUGS))}.\n"
    "Guidelines:\n"
    "- dairy: milk, cheese, butter, cream, yogurt, whey\n"
    "- poultry: chicken, turkey, duck; includes poultry soup/stew\n"
    "- red_meat: beef, lamb, veal, game; not poultry or pork\n"
    "- pork: pig/swine products\n"
    "- fish vs shellfish: finfish vs crustaceans/mollusks\n"
    "- tree_nut vs peanut: peanut is legume, not tree_nut\n"
    "- wheat: wheat/gluten grains (not rice, corn, oats alone)\n"
    "- root_vegetable: onion, garlic, potato, carrot, beet, ginger root\n"
    "- Plant foods (tomato, rice, olive oil, fruit juice) usually have none.\n"
    "- Process/specification classes should have none.\n"
    "Output JSON only."
)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _format_item_line(it: dict[str, str]) -> str:
    line = f"- {it['foodon_id']} | {it['label']}"
    ctx = it.get("ontology_context")
    if ctx:
        line += f" | tree: {ctx}"
    return line


def _build_batch_prompt(items: list[dict[str, str]]) -> str:
    lines = [_format_item_line(it) for it in items]
    return (
        "Classify each FoodOn class. Return JSON:\n"
        '{"items": [{"foodon_id": "FOODON_...", "contains": ["dairy"], '
        '"confidence": 0.0-1, "rationale": "<=15 words"}]}\n'
        "Use an empty contains list when no restrictions apply.\n"
        "Tree context (parents/siblings) is advisory; prefer the class label and "
        "FoodOn placement over guessing.\n\n"
        "Items:\n" + "\n".join(lines)
    )


def _parse_batch_response(
    content: str,
    items: list[dict[str, str]],
) -> list[dict[str, Any]]:
    valid_ids = {it["foodon_id"] for it in items}
    by_id = {it["foodon_id"]: it["label"] for it in items}
    try:
        parsed = _extract_json(content)
    except json.JSONDecodeError:
        return [
            {
                "foodon_id": it["foodon_id"],
                "label": it["label"],
                "contains": [],
                "confidence": 0.0,
                "rationale": "judge_parse_error",
                "error": True,
            }
            for it in items
        ]

    raw_items = parsed.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    out_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_items:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("foodon_id") or "")
        if node_id not in valid_ids:
            continue
        contains_raw = row.get("contains") or []
        if isinstance(contains_raw, str):
            contains_raw = [contains_raw]
        contains = sorted({str(s) for s in contains_raw if str(s) in VALID_SLUGS})
        out_by_id[node_id] = {
            "foodon_id": node_id,
            "label": by_id[node_id],
            "contains": contains,
            "confidence": float(row.get("confidence") or 0.0),
            "rationale": str(row.get("rationale") or ""),
            "error": False,
        }

    results: list[dict[str, Any]] = []
    for it in items:
        node_id = it["foodon_id"]
        if node_id in out_by_id:
            results.append(out_by_id[node_id])
        else:
            results.append(
                {
                    "foodon_id": node_id,
                    "label": it["label"],
                    "contains": [],
                    "confidence": 0.0,
                    "rationale": "missing_in_response",
                    "error": True,
                }
            )
    return results


class FoodOnContainsLLMJudge:
    """Batch LLM classifier for FoodOn contains dimensions."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api: Literal["openai", "ollama"] = "openai",
        temperature: float = 0.0,
        timeout_s: int = 180,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def classify_batch(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        if not items:
            return []
        user_prompt = _build_batch_prompt(items)
        try:
            if self.api == "ollama":
                content = self._ollama_chat(user_prompt)
            else:
                content = self._openai_chat(user_prompt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            return [
                {
                    "foodon_id": it["foodon_id"],
                    "label": it["label"],
                    "contains": [],
                    "confidence": 0.0,
                    "rationale": f"judge_error: {exc}",
                    "error": True,
                }
                for it in items
            ]
        results = _parse_batch_response(content, items)
        if any(r.get("error") for r in results) and len(items) > 1:
            # Truncated or malformed batch JSON — fall back to singles.
            out: list[dict[str, Any]] = []
            for it in items:
                out.extend(self.classify_batch([it]))
            return out
        return results

    def _openai_chat(self, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        if not self.base_url.endswith("/v1"):
            url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    def _ollama_chat(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body.get("message", {}).get("content", ""))
