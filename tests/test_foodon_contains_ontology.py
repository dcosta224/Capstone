"""Tests for ontology-aware LLM contains validation."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foodon_web"))

from foodon_contains_core import build_contains_lookup, lookup_contains
from foodon_contains_ontology import evaluate_llm_suggestions, evaluate_slug_suggestion


class _OntologyIndex:
    def __init__(self) -> None:
        self.labels = {
            "FOODON_FISH": "fish food product",
            "FOODON_CATFISH": "piece of catfish meat",
            "FOODON_TILAPIA": "piece of tilapia meat",
            "FOODON_EURO": "eurocode 0201 - fish category",
            "FOODON_PLANT": "olive oil",
        }
        self.children = {
            "FOODON_FISH": ["FOODON_CATFISH", "FOODON_TILAPIA"],
            "FOODON_CATFISH": [],
            "FOODON_TILAPIA": [],
            "FOODON_EURO": [],
            "FOODON_PLANT": [],
        }
        self.parents = {
            "FOODON_FISH": [],
            "FOODON_CATFISH": ["FOODON_FISH"],
            "FOODON_TILAPIA": ["FOODON_FISH"],
            "FOODON_EURO": [],
            "FOODON_PLANT": [],
        }

    def ancestry_path(self, node_id: str) -> list[str]:
        return list(self.parents.get(node_id, []))

    def matches_any_ancestor(self, node_id: str, ancestor_ids: tuple[str, ...]) -> bool:
        if node_id in ancestor_ids:
            return True
        return any(a in self.ancestry_path(node_id) for a in ancestor_ids)


def _table_for(index: _OntologyIndex, monkeypatch) -> pd.DataFrame:
    from diet_tags_core import ContainsTrigger, DietTagsRegistry

    fake_registry = DietTagsRegistry(
        universal=frozenset(),
        contains={
            "fish": ContainsTrigger("fish", ("catfish", "tilapia", "salmon"), ("FOODON_FISH",)),
            "poultry": ContainsTrigger("poultry", ("chicken",), ()),
        },
        nutrients={},
        ingredient_tags={},
        recipe_tags={},
    )
    monkeypatch.setattr("foodon_contains_core.load_diet_tags", lambda path=None: fake_registry)
    monkeypatch.setattr("foodon_contains_ontology.load_diet_tags", lambda path=None: fake_registry)
    df, _ = build_contains_lookup(index, foodon_only=True)
    return df


def test_piece_of_meat_high_conf_confirms_off_branch(monkeypatch):
    index = _OntologyIndex()
    table = _table_for(index, monkeypatch)

    confirmed, rejected, verdicts = evaluate_llm_suggestions(
        node_id="FOODON_CATFISH",
        label="piece of catfish meat",
        current=set(),
        llm_contains={"fish"},
        llm_confidence=0.95,
        foodon_index=index,
        contains_table=table,
    )
    assert confirmed == ["fish"]
    assert rejected == []
    assert "piece_of_meat_high_conf" in verdicts[0].reasons


def test_piece_of_beefalo_confirms_red_meat(monkeypatch):
    index = _OntologyIndex()
    index.labels["FOODON_BEEFALO"] = "piece of beefalo meat"
    index.children["FOODON_BEEFALO"] = []
    index.parents["FOODON_BEEFALO"] = []
    table = _table_for(index, monkeypatch)

    confirmed, rejected, _ = evaluate_llm_suggestions(
        node_id="FOODON_BEEFALO",
        label="piece of beefalo meat",
        current=set(),
        llm_contains={"red_meat"},
        llm_confidence=0.9,
        foodon_index=index,
        contains_table=table,
    )
    assert confirmed == ["red_meat"]
    assert rejected == []


def test_piece_of_meat_rule_requires_high_confidence(monkeypatch):
    index = _OntologyIndex()
    index.labels["FOODON_OFF"] = "piece of swordfish meat"
    index.children["FOODON_OFF"] = []
    index.parents["FOODON_OFF"] = []
    table = _table_for(index, monkeypatch)

    confirmed, rejected, _ = evaluate_llm_suggestions(
        node_id="FOODON_OFF",
        label="piece of swordfish meat",
        current=set(),
        llm_contains={"fish"},
        llm_confidence=0.8,
        foodon_index=index,
        contains_table=table,
    )
    assert confirmed == []
    assert rejected == ["fish"]


def test_confirms_fish_piece_with_label_and_parent_support(monkeypatch):
    index = _OntologyIndex()
    table = _table_for(index, monkeypatch)

    confirmed, rejected, verdicts = evaluate_llm_suggestions(
        node_id="FOODON_CATFISH",
        label="piece of catfish meat",
        current=set(),
        llm_contains={"fish"},
        llm_confidence=0.9,
        foodon_index=index,
        contains_table=table,
    )
    assert confirmed == ["fish"]
    assert rejected == []
    assert verdicts[0].label_match
    assert verdicts[0].under_configured_ancestor


def test_rejects_taxonomy_category_without_structure(monkeypatch):
    index = _OntologyIndex()
    table = _table_for(index, monkeypatch)

    confirmed, rejected, _ = evaluate_llm_suggestions(
        node_id="FOODON_EURO",
        label="eurocode 0201 - fish category",
        current=set(),
        llm_contains={"fish", "shellfish"},
        llm_confidence=0.95,
        foodon_index=index,
        contains_table=table,
    )
    assert confirmed == []
    assert set(rejected) == {"fish", "shellfish"}


def test_rejects_llm_only_without_label_or_tree_support(monkeypatch):
    index = _OntologyIndex()
    table = _table_for(index, monkeypatch)

    verdict = evaluate_slug_suggestion(
        node_id="FOODON_PLANT",
        slug="fish",
        label="olive oil",
        llm_confidence=0.9,
        foodon_index=index,
        contains_table=table,
    )
    assert not verdict.confirmed
    assert not verdict.label_match
    assert not verdict.under_configured_ancestor


def test_sibling_support_confirms_second_fish_piece(monkeypatch):
    index = _OntologyIndex()
    table = _table_for(index, monkeypatch)
    batch_tags = {"FOODON_CATFISH": {"fish"}}

    confirmed, rejected, _ = evaluate_llm_suggestions(
        node_id="FOODON_TILAPIA",
        label="piece of tilapia meat",
        current=set(),
        llm_contains={"fish"},
        llm_confidence=0.85,
        foodon_index=index,
        contains_table=table,
        llm_batch_tags=batch_tags,
    )
    assert confirmed == ["fish"]
    assert rejected == []
