#!/usr/bin/env python3
"""Generate notebooks/colab_ingredient_resolution_oss.ipynb (run once locally)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_ingredient_resolution_oss.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


OSS_PROBE_AND_MODEL = r'''
import json
import os
import re
import threading
import asyncio
from dataclasses import dataclass, asdict
from typing import Any

import psutil
import torch

PROBE: dict[str, Any] = {}


def _gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def probe_memory() -> dict[str, Any]:
    """Set judge concurrency and embedding batch from GPU VRAM + system RAM."""
    vram = _gpu_vram_gb()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    cfg: dict[str, Any] = {
        "gpu_name": gpu_name,
        "vram_gb": round(vram, 1),
        "ram_gb": round(ram_gb, 1),
        "backend": "transformers",
        "judge_concurrency": 2,
        "vllm_max_num_seqs": 0,
        "max_new_tokens_judge": 512,
        "max_new_tokens_short": 256,
        "embed_batch": 0,
    }

    # Prefer vLLM on A100-class GPUs
    if vram >= 38:
        cfg["backend"] = "vllm"
        cfg["vllm_max_num_seqs"] = 8
        cfg["judge_concurrency"] = 8
        if vram >= 70:
            cfg["vllm_max_num_seqs"] = 12
            cfg["judge_concurrency"] = 12
    elif vram >= 14:
        cfg["backend"] = "transformers"
        cfg["judge_concurrency"] = 4
    else:
        cfg["backend"] = "transformers"
        cfg["judge_concurrency"] = 2

    if ram_gb >= 50:
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        cfg["embed_batch"] = min(512, max(128, int(free_gb * 48)))
    elif ram_gb >= 25:
        cfg["embed_batch"] = 256
    else:
        cfg["embed_batch"] = 0  # use pre-built S3 embeddings only

    return cfg


PROBE = probe_memory()
print(json.dumps(PROBE, indent=2))


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(m.group(0))


@dataclass
class GenResult:
    parsed: dict[str, Any]
    raw: str
    prompt_tokens: int
    completion_tokens: int


class OssJsonLlm:
    """Open-source JSON LLM for judge / enrichment / portion pick."""

    def __init__(self, model_id: str, probe: dict[str, Any]):
        self.model_id = model_id
        self.probe = probe
        self.backend = probe.get("backend", "transformers")
        self._lock = threading.Lock()
        self._llm = None
        self._model = None
        self._tokenizer = None
        self._load()

    def _vllm_model_id(self) -> str:
        """Prefer AWQ weights for vLLM (fits A100 40GB with batching)."""
        if self.model_id.endswith("-AWQ"):
            return self.model_id
        if "14B" in self.model_id:
            return "Qwen/Qwen2.5-14B-Instruct-AWQ"
        if "7B" in self.model_id:
            return "Qwen/Qwen2.5-7B-Instruct-AWQ"
        return self.model_id

    def _load(self) -> None:
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams  # noqa: F401

                max_seqs = int(self.probe.get("vllm_max_num_seqs", 8))
                vllm_id = self._vllm_model_id()
                self._llm = LLM(
                    model=vllm_id,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.90,
                    max_model_len=8192,
                    max_num_seqs=max_seqs,
                    dtype="auto",
                )
                self._vllm_model_id_loaded = vllm_id
                self._SamplingParams = SamplingParams
                print(f"Loaded vLLM {vllm_id} max_num_seqs={max_seqs}")
                return
            except Exception as exc:
                print(f"vLLM load failed ({exc}); falling back to transformers")
                self.backend = "transformers"

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"Loaded transformers 4-bit {self.model_id}")

    def _chat_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.backend == "vllm":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        return len(tok.encode(text))

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        max_new_tokens: int = 512,
        extra_user: str | None = None,
    ) -> GenResult:
        full_user = user + (f"\n\n{extra_user}" if extra_user else "")
        full_user += "\n\nRespond with a single JSON object only. No markdown fences."
        prompt = self._chat_prompt(system, full_user)
        prompt_tokens = self._count_tokens(prompt)

        if self.backend == "vllm":
            params = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            outs = self._llm.generate([prompt], params)
            raw = outs[0].outputs[0].text.strip()
            completion_tokens = self._count_tokens(raw)
            return GenResult(extract_json(raw), raw, prompt_tokens, completion_tokens)

        def _gen() -> str:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        with self._lock:
            raw = _gen()
        completion_tokens = self._count_tokens(raw)
        return GenResult(extract_json(raw), raw, prompt_tokens, completion_tokens)

    async def generate_json_async(self, *args, **kwargs) -> GenResult:
        return await asyncio.to_thread(self.generate_json, *args, **kwargs)

    def meta(self) -> dict[str, Any]:
        meta = {
            "model_id": self.model_id,
            "backend": self.backend,
            "probe": self.probe,
        }
        if hasattr(self, "_vllm_model_id_loaded"):
            meta["vllm_model_id"] = self._vllm_model_id_loaded
        return meta


DEFAULT_MODEL = os.environ.get("OSS_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"

try:
    OSS_LLM = OssJsonLlm(DEFAULT_MODEL, PROBE)
except Exception as exc:
    print(f"OOM loading {DEFAULT_MODEL}: {exc}; trying {FALLBACK_MODEL}")
    PROBE["judge_concurrency"] = min(PROBE.get("judge_concurrency", 4), 4)
    OSS_LLM = OssJsonLlm(FALLBACK_MODEL, PROBE)

# Warmup: 3 short JSON calls
for i in range(3):
    r = OSS_LLM.generate_json(
        "You output JSON only.",
        f'Return {{"ok": true, "n": {i}}}',
        max_new_tokens=64,
    )
    print("warmup", i, r.parsed)

print("OSS backend ready:", json.dumps(OSS_LLM.meta(), indent=2))
'''.strip() + "\n"


PATCH_AND_RUN = r'''
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import ingredient_match_llm as iml
import line_enrichment_llm as lel
import portion_resolve_llm as prl
import openai_fallback
import sample_recipes
from portion_pipeline_feasibility import run_feasibility

CACHE = Path("/content/capstone_cache")
OUT = Path("/content/capstone_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True, exist_ok=True)
RUN_ID = OUT.name

os.environ["FOOD_4MACRO_CACHE"] = str(CACHE / "food_4macro.csv")
os.environ["CAPSTONE_RECIPE_CACHE"] = str(CACHE / "recipe_cache")

recipe_csv = CACHE / "RecipeNLG.csv"
sample_recipes.DEFAULT_RECIPE_CSV = recipe_csv

# Dummy OpenAI clients (unused after patches)
class _Dummy:
    class chat:
        class completions:
            @staticmethod
            async def create(**kwargs):
                raise RuntimeError("OpenAI client should not be called")


openai_fallback.get_async_openai_client = lambda: _Dummy()
openai_fallback.get_sync_openai_client = lambda: _Dummy()


async def oss_judge_async(
    client,
    model: str,
    user_prompt: str,
    valid_fdc_ids: set[int],
    *,
    system_prompt: str | None = None,
):
    sys_prompt = system_prompt or iml.SYSTEM_PROMPT
    prompt_tokens = completion_tokens = 0
    error = None
    parsed: dict = {}
    raw_response = None

    try:
        r = await OSS_LLM.generate_json_async(
            sys_prompt,
            user_prompt,
            max_new_tokens=PROBE.get("max_new_tokens_judge", 512),
        )
        parsed, raw_response = r.parsed, r.raw
        prompt_tokens += r.prompt_tokens
        completion_tokens += r.completion_tokens
        fdc_id = parsed.get("fdc_id")
        if fdc_id is not None and int(fdc_id) not in valid_fdc_ids:
            hint = (
                "Your previous fdc_id was not in the candidate list. "
                "Choose only from the listed fdc_id values, or null."
            )
            r2 = await OSS_LLM.generate_json_async(
                sys_prompt,
                user_prompt,
                max_new_tokens=PROBE.get("max_new_tokens_judge", 512),
                extra_user=hint,
            )
            parsed, raw_response = r2.parsed, r2.raw
            prompt_tokens += r2.prompt_tokens
            completion_tokens += r2.completion_tokens
            fdc_id = parsed.get("fdc_id")
            if fdc_id is not None and int(fdc_id) not in valid_fdc_ids:
                error = "invalid_fdc_id"
                parsed["fdc_id"] = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    matched_portion_id = parsed.get("matched_portion_id")
    if matched_portion_id is not None:
        try:
            matched_portion_id = int(matched_portion_id)
        except (TypeError, ValueError):
            matched_portion_id = None

    return {
        "fdc_id": parsed.get("fdc_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "matched_portion_id": matched_portion_id,
        "negligible_calories": bool(parsed.get("negligible_calories", False)),
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


async def oss_enrich_one_async(client, model: str, ingredient: str, rules_plan):
    from line_enrichment_llm import build_user_prompt, validate_response

    prompt_tokens = completion_tokens = 0
    error = None
    parsed: dict = {}
    raw_response = None

    try:
        user = build_user_prompt(ingredient, rules_plan)
        r = await OSS_LLM.generate_json_async(
            lel.SYSTEM_PROMPT,
            user,
            max_new_tokens=PROBE.get("max_new_tokens_short", 256),
        )
        parsed, raw_response = r.parsed, r.raw
        prompt_tokens += r.prompt_tokens
        completion_tokens += r.completion_tokens
        validation_error = validate_response(parsed)
        if validation_error:
            r2 = await OSS_LLM.generate_json_async(
                lel.SYSTEM_PROMPT,
                user,
                max_new_tokens=PROBE.get("max_new_tokens_short", 256),
                extra_user=f"Validation failed ({validation_error}). Fix and resubmit.",
            )
            parsed, raw_response = r2.parsed, r2.raw
            prompt_tokens += r2.prompt_tokens
            completion_tokens += r2.completion_tokens
            if validate_response(parsed):
                error = validation_error
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    from ingredient_parse_llm import normalize_ingredient_key

    return {
        "ingredient_norm": normalize_ingredient_key(ingredient),
        "ingredient_raw": ingredient,
        "enrichment": parsed if not error else {},
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "error": error,
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "price_estimate_usd": 0.0,
    }


def oss_pick_portion_sync(
    model: str,
    *,
    ingredient: str,
    quantity: float,
    unit: str | None,
    name: str | None,
    amount_kind: str,
    fdc_id: int,
    raw_rows: list,
):
    block = prl.format_portion_options(raw_rows)
    user = prl.build_user_prompt(
        ingredient=ingredient,
        quantity=quantity,
        unit=unit,
        name=name,
        amount_kind=amount_kind,
        fdc_id=fdc_id,
        portion_block=block,
    )
    r = OSS_LLM.generate_json(
        prl.SYSTEM_PROMPT,
        user,
        max_new_tokens=PROBE.get("max_new_tokens_short", 256),
    )
    parsed = r.parsed
    return {
        "portion_id": parsed.get("portion_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "price_estimate_usd": 0.0,
        "user_prompt": user,
    }


iml.judge_async = oss_judge_async
lel.enrich_one_async = oss_enrich_one_async
prl.pick_portion_sync = oss_pick_portion_sync

limit = int(os.environ["COLAB_LIMIT"]) if os.environ.get("COLAB_LIMIT") else None
model_label = OSS_LLM.model_id

report = run_feasibility(
    n_recipes=1000,
    seed=42,
    model=model_label,
    out_dir=OUT,
    baseline_dir=None,
    food_cache_dir=CACHE / "food_cache",
    limit=limit,
    concurrency=int(PROBE.get("judge_concurrency", 4)),
    skip_portion_llm=False,
    force_amount=False,
    force_judging=False,
    force_payloads=False,
    force_all=False,
    finalize_only=False,
    only_no_portion=False,
    use_mlflow=False,
    sample_manifest=CACHE / "sampled_recipe_ids.json",
    recipe_cache_dir=CACHE / "recipe_cache",
)

(OUT / "oss_model_meta.json").write_text(
    json.dumps({**OSS_LLM.meta(), "run_id": RUN_ID, "colab_limit": limit}, indent=2) + "\n"
)
print("Run complete:", OUT)
print(json.dumps({k: report[k] for k in (
    "n_lines", "fdc_match_rate_all", "gram_resolve_rate_all",
    "fdc_and_gram_rate_all", "judge_error_count", "elapsed_sec",
) if k in report}, indent=2))
'''.strip() + "\n"


cells = [
    md(
        "# Colab OSS ingredient resolution pipeline\n\n"
        "Runs the full `portion_pipeline_feasibility` pipeline on the canonical **1,000 recipes** "
        "(seed 42) using an open-source Hugging Face model instead of OpenAI.\n\n"
        "**Prerequisites:** Colab GPU runtime (A100 recommended), Supabase credentials, AWS S3 access, "
        "and the input bundle at `s3://{artifacts}/colab/feasibility_1000_seed42/`.\n\n"
        "See [`docs/COLAB_OSS_RESOLUTION.md`](../docs/COLAB_OSS_RESOLUTION.md) for setup."
    ),
    code("!nvidia-smi\n!free -h"),
    code(
        "%%capture\n"
        "!pip install -q torch transformers accelerate bitsandbytes vllm sentence-transformers \\\n"
        "  pandas pyarrow psycopg2-binary rapidfuzz ingredient-parser-nlp \\\n"
        "  faiss-cpu scikit-learn tqdm psutil networkx python-dotenv"
    ),
    code(
        "import os\n"
        "from google.colab import userdata\n\n"
        "def _secret(key: str, default: str = '') -> str:\n"
        "    try:\n"
        "        return userdata.get(key)\n"
        "    except Exception:\n"
        "        return os.environ.get(key, default)\n\n"
        "for key in (\n"
        "    'PG_POOL_USER', 'PG_PASSWORD', 'PG_POOL_HOST', 'PG_POOL_SESSION_PORT',\n"
        "    'PG_DATABASE', 'PG_SSL_MODE', 'HF_TOKEN',\n"
        "    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION',\n"
        "    'S3_BUCKET_ARTIFACTS', 'S3_BUCKET_RAW',\n"
        "):\n"
        "    val = _secret(key)\n"
        "    if val:\n"
        "        os.environ[key] = val\n\n"
        "os.environ.setdefault('PG_POOL_SESSION_PORT', '5432')\n"
        "os.environ.setdefault('PG_DATABASE', 'postgres')\n"
        "os.environ.setdefault('PG_SSL_MODE', 'require')\n"
        "os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')\n"
        "os.environ.setdefault('OSS_MODEL_ID', 'Qwen/Qwen2.5-14B-Instruct')\n"
        "# Optional smoke test: os.environ['COLAB_LIMIT'] = '100'\n"
        "print('Secrets loaded. S3 bucket:', os.environ.get('S3_BUCKET_ARTIFACTS'))"
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n\n"
        "REPO = Path('/content/Capstone')\n"
        "if not REPO.is_dir():\n"
        "    !git clone https://github.com/dcosta224/Capstone.git /content/Capstone\n"
        "%cd /content/Capstone\n"
        "sys.path.insert(0, str(REPO / 'scripts'))\n"
        "print('Repo ready:', REPO)"
    ),
    code(
        "import os\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        "CACHE = Path('/content/capstone_cache')\n"
        "CACHE.mkdir(parents=True, exist_ok=True)\n"
        "bucket = os.environ['S3_BUCKET_ARTIFACTS']\n"
        "raw_bucket = os.environ.get('S3_BUCKET_RAW', '')\n\n"
        "subprocess.run([\n"
        "    'aws', 's3', 'sync',\n"
        "    f's3://{bucket}/colab/feasibility_1000_seed42/', str(CACHE) + '/',\n"
        "], check=True)\n\n"
        "recipe_csv = CACHE / 'RecipeNLG.csv'\n"
        "if not recipe_csv.is_file():\n"
        "    if not raw_bucket:\n"
        "        raise SystemExit('Set S3_BUCKET_RAW or place RecipeNLG.csv in the cache')\n"
        "    subprocess.run([\n"
        "        'aws', 's3', 'cp',\n"
        "        f's3://{raw_bucket}/Data/recipes/RecipeNLG.csv', str(recipe_csv),\n"
        "    ], check=True)\n"
        "print('Cache:', list(CACHE.iterdir())[:12], '...')"
    ),
    code(OSS_PROBE_AND_MODEL),
    code(PATCH_AND_RUN),
    code(
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        "bucket = os.environ['S3_BUCKET_ARTIFACTS']\n"
        "run_dir = OUT\n"
        "dest = f's3://{bucket}/colab/runs/{RUN_ID}/'\n"
        "subprocess.run(['aws', 's3', 'sync', str(run_dir) + '/', dest], check=True)\n"
        "print('Uploaded to', dest)"
    ),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "baseline = json.loads((CACHE / 'baseline_summary.json').read_text())\n"
        "new_report = json.loads((OUT / 'feasibility_report.json').read_text())\n\n"
        "keys = [\n"
        "    'fdc_match_rate_all', 'gram_resolve_rate_all', 'fdc_and_gram_rate_all',\n"
        "    'fdc_and_gram_rate_needs_portion', 'judge_error_count', 'n_lines',\n"
        "]\n"
        "print('metric | baseline (GPT) | OSS')\n"
        "for k in keys:\n"
        "    b = baseline.get(k)\n"
        "    n = new_report.get(k)\n"
        "    print(f'{k}: {b} -> {n}')"
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"Wrote {OUT}")
