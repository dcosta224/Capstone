"""Tests for foodon_contains_judge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from foodon_contains_judge import _parse_batch_response


def test_parse_batch_response():
    items = [
        {"foodon_id": "FOODON_A", "label": "chicken stew or hash"},
        {"foodon_id": "FOODON_B", "label": "olive oil"},
    ]
    content = """{
      "items": [
        {"foodon_id": "FOODON_A", "contains": ["poultry"], "confidence": 0.9, "rationale": "chicken stew"},
        {"foodon_id": "FOODON_B", "contains": [], "confidence": 0.95, "rationale": "plant oil"}
      ]
    }"""
    out = _parse_batch_response(content, items)
    assert out[0]["contains"] == ["poultry"]
    assert out[1]["contains"] == []
