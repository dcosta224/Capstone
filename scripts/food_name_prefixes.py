#!/usr/bin/env python3
"""Extract leading word spans before commas, prepositions, or conjunctions."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "scratch" / "food_mvp_volume_gram_weight.txt"
DEFAULT_OUT = ROOT / "scratch" / "food_mvp_volume_gram_weight_prefixes.txt"

# Word-boundary prepositions and conjunctions (exclude "&" — common in food names).
_PREP_CONJ = (
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "among",
    "around",
    "as",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "but",
    "by",
    "despite",
    "down",
    "during",
    "except",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "like",
    "near",
    "nor",
    "of",
    "off",
    "on",
    "onto",
    "or",
    "out",
    "outside",
    "over",
    "past",
    "since",
    "so",
    "than",
    "that",
    "though",
    "through",
    "to",
    "toward",
    "under",
    "until",
    "up",
    "upon",
    "via",
    "when",
    "where",
    "whether",
    "while",
    "with",
    "within",
    "without",
    "yet",
    "and",
    "although",
    "because",
    "if",
    "unless",
    "whereas",
    "while",
)
_PREP_CONJ_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(w) for w in _PREP_CONJ) + r")(?!\w)",
    re.IGNORECASE,
)


def words_before_delimiters(text: str) -> str:
    """Return all words before the first comma, preposition, or conjunction.

    If none appear, returns the full trimmed line. Strips trailing commas and
    whitespace from the prefix.
    """
    text = text.strip()
    if not text:
        return ""

    positions: list[int] = []
    if (comma := text.find(",")) >= 0:
        positions.append(comma)
    if (match := _PREP_CONJ_RE.search(text)):
        positions.append(match.start())

    if not positions:
        return text

    prefix = text[: min(positions)].rstrip(" ,")
    return prefix


def distinct_case_insensitive(values: list[str]) -> list[str]:
    """Keep one spelling per case-insensitive key (first seen wins)."""
    seen: dict[str, str] = {}
    for value in values:
        value = value.strip()
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            seen[key] = value
    return sorted(seen.values(), key=str.casefold)


def write_prefix_names_file(
    input_path: Path = DEFAULT_IN,
    output_path: Path = DEFAULT_OUT,
) -> int:
    """Read one name per line; write case-insensitively distinct prefixes (sorted)."""
    lines = input_path.read_text(encoding="utf-8").splitlines()
    prefixes = distinct_case_insensitive(
        [words_before_delimiters(line) for line in lines]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(prefixes) + "\n", encoding="utf-8")
    return len(prefixes)


def main() -> None:
    count = write_prefix_names_file()
    print(f"Wrote {count:,} prefixes → {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
