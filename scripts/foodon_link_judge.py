"""Optional local LLM judge for FoodOn candidate reranking (Ollama or OpenAI-compatible)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Literal


SYSTEM_PROMPT = (
    "You map a USDA food description to the best FoodOn ontology class for nutrition tagging.\n"
    "Choose exactly one foodon_id from the candidate list, or abstain if none fit.\n"
    "Output JSON only with keys: foodon_id (string or null), confidence (0-1), rationale (<=20 words)."
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


def _build_user_prompt(description: str, candidates: list[dict[str, Any]]) -> str:
    lines = [
        f"{i + 1}. {c['id']} | {c.get('label', '')} | score={float(c.get('score', 0)):.3f}"
        for i, c in enumerate(candidates[:20])
    ]
    return (
        f"USDA description: {description}\n\n"
        "Candidates:\n"
        + "\n".join(lines)
        + '\n\nReturn JSON: {"foodon_id": "..." or null, "confidence": 0-1, "rationale": "..."}'
    )


def _parse_judge_response(
    content: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_ids = {str(c["id"]) for c in candidates}
    try:
        parsed = _extract_json(content)
    except json.JSONDecodeError:
        return {"abstain": True, "confidence": 0.0, "rationale": "judge_parse_error"}

    foodon_id = parsed.get("foodon_id")
    if foodon_id is not None:
        foodon_id = str(foodon_id)
        if foodon_id not in valid_ids:
            return {
                "abstain": True,
                "confidence": 0.0,
                "rationale": "invalid_foodon_id",
            }
        label = next((c.get("label") for c in candidates if str(c["id"]) == foodon_id), None)
        return {
            "foodon_id": foodon_id,
            "foodon_label": label,
            "confidence": float(parsed.get("confidence") or 0.0),
            "rationale": str(parsed.get("rationale") or ""),
        }

    return {
        "abstain": True,
        "confidence": float(parsed.get("confidence") or 0.0),
        "rationale": str(parsed.get("rationale") or "abstain"),
    }


class FoodOnLLMJudge:
    """LLM reranker with Ollama or OpenAI-compatible (LM Studio) backends."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api: Literal["openai", "ollama"] = "openai",
        temperature: float = 0.0,
        timeout_s: int = 180,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.temperature = temperature
        self.timeout_s = timeout_s

    def __call__(self, description: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        user_prompt = _build_user_prompt(description, candidates)
        try:
            if self.api == "ollama":
                content = self._ollama_chat(user_prompt)
            else:
                content = self._openai_chat(user_prompt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            return {"abstain": True, "confidence": 0.0, "rationale": f"judge_error: {exc}"}
        return _parse_judge_response(content, candidates)

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


# Back-compat alias
OllamaFoodOnJudge = FoodOnLLMJudge
