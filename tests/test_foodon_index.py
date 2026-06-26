"""Tests for foodon_index (no full OWL load)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foodon_web"))

from foodon_index import FoodOnIndex, compact_id


def _tiny_index() -> FoodOnIndex:
    payload = {
        "labels": {
            "FOODON_00000001": "mammalian milk product",
            "FOODON_00000002": "cheddar cheese",
            "FOODON_00000003": "wheat food product",
            "FOODON_00000004": "wheat flour food product",
        },
        "parents": {
            "FOODON_00000001": [],
            "FOODON_00000002": ["FOODON_00000001"],
            "FOODON_00000003": [],
            "FOODON_00000004": ["FOODON_00000003"],
        },
        "children": {
            "FOODON_00000001": ["FOODON_00000002"],
            "FOODON_00000002": [],
            "FOODON_00000003": ["FOODON_00000004"],
            "FOODON_00000004": [],
        },
        "roots": ["FOODON_00000001", "FOODON_00000003"],
        "descendant_counts": {
            "FOODON_00000001": 1,
            "FOODON_00000002": 0,
            "FOODON_00000003": 1,
            "FOODON_00000004": 0,
        },
        "label_keys": [
            "FOODON_00000001",
            "FOODON_00000002",
            "FOODON_00000003",
            "FOODON_00000004",
        ],
    }
    return FoodOnIndex(payload)


def test_compact_id_from_uri():
    uri = "http://purl.obolibrary.org/obo/FOODON_00001274"
    assert compact_id(uri) == "FOODON_00001274"


def test_is_descendant_of():
    index = _tiny_index()
    assert index.is_descendant_of("FOODON_00000002", "FOODON_00000001")
    assert not index.is_descendant_of("FOODON_00000004", "FOODON_00000001")


def test_matches_any_ancestor():
    index = _tiny_index()
    assert index.matches_any_ancestor("FOODON_00000002", ("FOODON_00000001",))
    assert index.matches_any_ancestor("FOODON_00000004", ("FOODON_00000003",))


def test_search_returns_cheddar():
    index = _tiny_index()
    hits = index.search("cheddar", limit=3)
    assert hits
    assert hits[0]["id"] == "FOODON_00000002"
