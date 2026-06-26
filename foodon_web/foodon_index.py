"""Build and query a searchable index over a local FoodOn OWL file."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import RDFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from foodon_paths import FOODON_INDEX_CACHE, FOODON_SYNONYMS_TSV, resolve_owl_path

try:
    from rapidfuzz import fuzz, process
except ImportError:
    process = None
    fuzz = None

OBO_PREFIX = "http://purl.obolibrary.org/obo/"
BFO_PREFIXES = ("BFO_", "CHEBI_", "NCBITaxon_", "UBERON_")

PREFERRED_ROOTS = (
    "BFO_0000040",  # material entity
)


def compact_id(uri: str | URIRef) -> str | None:
    text = str(uri)
    if text.startswith(OBO_PREFIX):
        return text[len(OBO_PREFIX) :].replace(":", "_")
    if "#" in text:
        frag = text.rsplit("#", 1)[-1]
        if frag:
            return frag.replace(":", "_")
    return None


def _is_bfo(node_id: str) -> bool:
    return node_id.startswith("BFO_")


class FoodOnIndex:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.labels: dict[str, str] = payload["labels"]
        self.parents: dict[str, list[str]] = payload["parents"]
        self.children: dict[str, list[str]] = payload["children"]
        self.roots: list[str] = payload["roots"]
        self._descendant_counts: dict[str, int] = payload.get("descendant_counts", {})
        self._label_keys: list[str] = payload.get("label_keys", list(self.labels.keys()))
        self._search_labels: list[str] = [
            self.labels[node_id].lower() for node_id in self._label_keys
        ]

    @classmethod
    def from_cache(cls, cache_path: Path) -> FoodOnIndex:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return cls(payload)

    @classmethod
    def from_owl(
        cls,
        *,
        owl_path: Path | None = None,
        synonyms_path: Path | None = None,
        cache_path: Path | None = None,
        force_rebuild: bool = False,
    ) -> FoodOnIndex:
        cache = cache_path or FOODON_INDEX_CACHE
        if cache.is_file() and not force_rebuild:
            return cls.from_cache(cache)

        payload = cls._build_payload(
            owl_path or resolve_owl_path(),
            synonyms_path or FOODON_SYNONYMS_TSV,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
        return cls(payload)

    @classmethod
    def _build_payload(cls, owl_path: Path, synonyms_path: Path) -> dict[str, Any]:
        graph = Graph()
        graph.parse(owl_path, format="xml")

        labels: dict[str, str] = {}
        for subj, _pred, obj in graph.triples((None, RDFS.label, None)):
            node_id = compact_id(subj)
            if not node_id:
                continue
            label = str(obj).strip()
            if not label:
                continue
            if node_id not in labels or len(label) < len(labels[node_id]):
                labels[node_id] = label

        if synonyms_path.is_file():
            cls._merge_synonyms_tsv(synonyms_path, labels)

        parents: dict[str, list[str]] = {node_id: [] for node_id in labels}
        children: dict[str, list[str]] = {node_id: [] for node_id in labels}

        for subj, _pred, obj in graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(obj, BNode):
                continue
            child_id = compact_id(subj)
            parent_id = compact_id(obj)
            if not child_id or not parent_id:
                continue
            if child_id not in labels or parent_id not in labels:
                continue
            if parent_id not in parents[child_id]:
                parents[child_id].append(parent_id)
            if child_id not in children[parent_id]:
                children[parent_id].append(child_id)

        roots = sorted(
            node_id for node_id, ps in parents.items() if not ps and node_id in labels
        )
        descendant_counts = cls._compute_descendant_counts(children)

        return {
            "labels": labels,
            "parents": parents,
            "children": children,
            "roots": roots,
            "descendant_counts": descendant_counts,
            "label_keys": sorted(labels.keys()),
            "owl_path": str(owl_path),
        }

    @staticmethod
    def _merge_synonyms_tsv(path: Path, labels: dict[str, str]) -> None:
        line_re = re.compile(
            r"^<([^>]+)>\s+.*?\t\"label\"\t\"(.+?)\"(?:@en)?\s*$"
        )
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = line_re.match(line)
            if not m:
                continue
            node_id = compact_id(m.group(1))
            label = m.group(2).strip()
            if node_id and label and node_id not in labels:
                labels[node_id] = label

    @staticmethod
    def _compute_descendant_counts(children: dict[str, list[str]]) -> dict[str, int]:
        counts: dict[str, int] = {}

        def count(node_id: str, visiting: set[str]) -> int:
            if node_id in counts:
                return counts[node_id]
            if node_id in visiting:
                return 0
            visiting.add(node_id)
            total = 0
            for child in children.get(node_id, []):
                total += 1 + count(child, visiting)
            visiting.remove(node_id)
            counts[node_id] = total
            return total

        for node_id in children:
            count(node_id, set())
        return counts

    def preferred_roots(self) -> list[str]:
        out = [node_id for node_id in PREFERRED_ROOTS if node_id in self.labels]
        if out:
            return out
        return self.roots[:5]

    def ancestry_path(self, node_id: str) -> list[str]:
        path: list[str] = []
        seen: set[str] = set()
        frontier = list(self.parents.get(node_id, []))
        while frontier:
            parent = frontier.pop(0)
            if parent in seen:
                continue
            seen.add(parent)
            path.append(parent)
            frontier.extend(self.parents.get(parent, []))
        return path

    def is_descendant_of(self, node_id: str, ancestor_id: str) -> bool:
        if node_id == ancestor_id:
            return True
        return ancestor_id in self.ancestry_path(node_id)

    def matches_any_ancestor(self, node_id: str, ancestor_ids: tuple[str, ...]) -> bool:
        return any(self.is_descendant_of(node_id, anc) for anc in ancestor_ids)

    def node_summary(self, node_id: str) -> dict[str, Any] | None:
        if node_id not in self.labels:
            return None
        return {
            "id": node_id,
            "label": self.labels[node_id],
            "is_bfo": _is_bfo(node_id),
            "has_children": bool(self.children.get(node_id)),
            "descendant_count": int(self._descendant_counts.get(node_id, 0)),
        }

    def child_summaries(self, node_id: str) -> list[dict[str, Any]]:
        kids = sorted(
            self.children.get(node_id, []),
            key=lambda cid: self.labels.get(cid, "").lower(),
        )
        summaries = [self.node_summary(cid) for cid in kids]
        return [s for s in summaries if s is not None]

    def search(self, query: str, *, limit: int = 25) -> list[dict[str, Any]]:
        q = query.strip().lower()
        if not q:
            return []

        if process is not None and fuzz is not None:
            hits = process.extract(
                q,
                self._search_labels,
                scorer=fuzz.WRatio,
                limit=limit,
            )
            results: list[dict[str, Any]] = []
            for label_text, score, idx in hits:
                if score < 55:
                    continue
                node_id = self._label_keys[idx]
                results.append(
                    {
                        "id": node_id,
                        "label": self.labels[node_id],
                        "score": round(score / 100.0, 3),
                        "descendant_count": int(self._descendant_counts.get(node_id, 0)),
                    }
                )
            return results

        # Fallback without rapidfuzz
        results = []
        for node_id, label in self.labels.items():
            if q in label.lower():
                results.append(
                    {
                        "id": node_id,
                        "label": label,
                        "score": 0.8,
                        "descendant_count": int(self._descendant_counts.get(node_id, 0)),
                    }
                )
        results.sort(key=lambda r: (-r["score"], r["label"]))
        return results[:limit]

    def best_match(self, text: str, *, min_score: float = 0.55) -> dict[str, Any] | None:
        hits = self.search(text, limit=1)
        if not hits:
            return None
        top = hits[0]
        if top["score"] < min_score:
            return None
        return top
