"""Tests for tiered FoodOn linking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from foodon_link_core import link_food_to_foodon


class _FakeFoodOnIndex:
    def __init__(self, match: dict | None) -> None:
        self._match = match

    def best_match(self, text: str, min_score: float = 0.55):
        if self._match is None:
            return None
        if self._match["score"] < min_score:
            return None
        return self._match


class _FakeEmbedIndex:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def search(self, text: str, k: int = 20):
        return self._hits[:k]


def test_auto_fuzzy_high_confidence():
    foodon = _FakeFoodOnIndex(
        {"id": "FOODON_A", "label": "cheddar cheese", "score": 0.92}
    )
    embed = _FakeEmbedIndex(
        [{"id": "FOODON_A", "label": "cheddar cheese", "score": 0.81}]
    )
    result = link_food_to_foodon(
        1,
        "Cheese, cheddar",
        foodon_index=foodon,
        embed_index=embed,
    )
    assert result.match_method == "auto_fuzzy"
    assert result.foodon_id == "FOODON_A"
    assert result.reviewed is False


def test_auto_semantic_when_fuzzy_weak():
    foodon = _FakeFoodOnIndex({"id": "FOODON_BAD", "label": "flour", "score": 0.60})
    embed = _FakeEmbedIndex(
        [{"id": "FOODON_RICE", "label": "rice flour", "score": 0.88}]
    )
    result = link_food_to_foodon(
        2,
        "Rice flour",
        foodon_index=foodon,
        embed_index=embed,
    )
    assert result.match_method == "auto_semantic"
    assert result.foodon_id == "FOODON_RICE"


def test_needs_review_low_confidence():
    foodon = _FakeFoodOnIndex({"id": "FOODON_X", "label": "food", "score": 0.62})
    embed = _FakeEmbedIndex([{"id": "FOODON_X", "label": "food", "score": 0.62}])
    result = link_food_to_foodon(
        3,
        "Natural flavor",
        foodon_index=foodon,
        embed_index=embed,
    )
    assert result.match_method == "needs_review"
    assert result.reviewed is True
