"""Tests for foodon_contains_core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foodon_web"))

from foodon_contains_core import build_contains_lookup, lookup_contains


class _MiniIndex:
    def __init__(self) -> None:
        self.labels = {
            "FOODON_DAIRY": "mammalian milk product",
            "FOODON_CHEESE": "cheese",
            "FOODON_CHEDDAR": "cheddar cheese",
            "FOODON_EGG": "egg food product",
            "FOODON_PLANT": "plant food",
        }
        self.children = {
            "FOODON_DAIRY": ["FOODON_CHEESE"],
            "FOODON_CHEESE": ["FOODON_CHEDDAR"],
            "FOODON_EGG": [],
            "FOODON_PLANT": [],
            "FOODON_CHEDDAR": [],
        }


def test_dairy_inherits_to_cheddar(monkeypatch):
    from diet_tags_core import ContainsTrigger, DietTagsRegistry

    fake_registry = DietTagsRegistry(
        universal=frozenset(),
        contains={
            "dairy": ContainsTrigger("dairy", (), ("FOODON_DAIRY",)),
            "egg": ContainsTrigger("egg", (), ("FOODON_EGG",)),
        },
        nutrients={},
        ingredient_tags={},
        recipe_tags={},
    )
    monkeypatch.setattr("foodon_contains_core.load_diet_tags", lambda path=None: fake_registry)

    df, summary = build_contains_lookup(_MiniIndex(), foodon_only=True)
    assert lookup_contains("FOODON_CHEDDAR", df) == {"dairy"}
    assert lookup_contains("FOODON_PLANT", df) == set()
    assert summary["tagged_counts"]["dairy"] == 3


def test_tree_nut_excludes_peanut(monkeypatch):
    from diet_tags_core import ContainsTrigger, DietTagsRegistry

    fake_registry = DietTagsRegistry(
        universal=frozenset(),
        contains={
            "tree_nut": ContainsTrigger(
                "tree_nut",
                (),
                ("FOODON_NUT",),
                ("FOODON_PEANUT",),
            ),
            "peanut": ContainsTrigger("peanut", (), ("FOODON_PEANUT",)),
        },
        nutrients={},
        ingredient_tags={},
        recipe_tags={},
    )
    monkeypatch.setattr("foodon_contains_core.load_diet_tags", lambda path=None: fake_registry)

    index = _MiniIndex()
    index.labels.update(
        {
            "FOODON_NUT": "nut food product",
            "FOODON_ALMOND": "almond",
            "FOODON_PEANUT": "peanut food product",
            "FOODON_RAW": "peanut",
        }
    )
    index.children.update(
        {
            "FOODON_NUT": ["FOODON_ALMOND", "FOODON_PEANUT"],
            "FOODON_PEANUT": ["FOODON_RAW"],
            "FOODON_ALMOND": [],
            "FOODON_RAW": [],
        }
    )

    df, _ = build_contains_lookup(index, foodon_only=True)
    assert lookup_contains("FOODON_ALMOND", df) == {"tree_nut"}
    assert lookup_contains("FOODON_RAW", df) == {"peanut"}
    assert "tree_nut" not in lookup_contains("FOODON_RAW", df)
