"""Load the 36-puzzle benchmark from puzzles/templates.json and puzzles/puzzles.json.

This module provides the single read-side entry point for the benchmark.
It joins each narrative puzzle variant in puzzles.json with its parent formal
template in templates.json and returns a flat list of dictionaries ready for
the experimental pipeline.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PUZZLES_DIR = Path(__file__).resolve().parent.parent / "puzzles"

EXPECTED_NUM_PUZZLES = 36
EXPECTED_NUM_TEMPLATES = 9
EXPECTED_VARIANTS = ("familiar", "belief_violating", "artificial", "abstract")

REQUIRED_PUZZLE_FIELDS = {
    "id",
    "template_id",
    "variant_type",
    "domain_label",
    "narrative",
    "question",
}

REQUIRED_TEMPLATE_FIELDS = {
    "template_id",
    "ground_truth",
    "query_type",
    "category",
    "axioms",
    "query",
}


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file expected to contain a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}")

    return data


def _check_required_fields(item: dict[str, Any], required: set[str], source: str) -> None:
    """Raise an error if a dictionary is missing required fields."""
    missing = required - set(item)

    if missing:
        identifier = item.get("id") or item.get("template_id") or "<unknown>"
        raise ValueError(
            f"{source} item {identifier!r} is missing required fields: {sorted(missing)}"
        )


def _check_unique(items: list[dict[str, Any]], key: str, source: str) -> None:
    """Raise an error if a key is duplicated across items."""
    seen = set()

    for item in items:
        value = item.get(key)

        if value in seen:
            raise ValueError(f"Duplicate {key!r} in {source}: {value!r}")

        seen.add(value)


def _load_templates(puzzles_dir: Path) -> dict[str, dict[str, Any]]:
    """Load templates.json and return a mapping from template_id to template."""
    templates_path = puzzles_dir / "templates.json"
    templates_data = _load_json(templates_path)

    if len(templates_data) != EXPECTED_NUM_TEMPLATES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TEMPLATES} templates, "
            f"but loaded {len(templates_data)}."
        )

    for template in templates_data:
        _check_required_fields(template, REQUIRED_TEMPLATE_FIELDS, "templates.json")

    _check_unique(templates_data, "template_id", "templates.json")

    return {template["template_id"]: template for template in templates_data}


def load_puzzles(puzzles_dir: Path = PUZZLES_DIR) -> list[dict[str, Any]]:
    """Return a flat list of puzzle dictionaries with template metadata joined in.

    Each returned dictionary contains all fields from the puzzle variant in
    puzzles.json, plus the following fields from the parent template:

        gold_label  : ground_truth from the parent template
        query_type  : entailment or satisfiability
        category    : template category
        axioms      : formal axiom list
        query       : formal query, e.g. "X1", "X2 ∧ X3", or "SAT(axioms)"
    """
    templates = _load_templates(puzzles_dir)

    puzzles_path = puzzles_dir / "puzzles.json"
    puzzles = _load_json(puzzles_path)

    if len(puzzles) != EXPECTED_NUM_PUZZLES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_PUZZLES} puzzles, but loaded {len(puzzles)}."
        )

    for puzzle in puzzles:
        _check_required_fields(puzzle, REQUIRED_PUZZLE_FIELDS, "puzzles.json")

    _check_unique(puzzles, "id", "puzzles.json")

    out: list[dict[str, Any]] = []
    variants_by_template: dict[str, set[str]] = defaultdict(set)

    for puzzle in puzzles:
        template_id = puzzle["template_id"]
        variant_type = puzzle["variant_type"]

        if template_id not in templates:
            raise ValueError(
                f"Puzzle {puzzle['id']!r} refers to unknown template_id: {template_id!r}"
            )

        if variant_type not in EXPECTED_VARIANTS:
            raise ValueError(
                f"Puzzle {puzzle['id']!r} has unexpected variant_type: {variant_type!r}"
            )

        template = templates[template_id]
        variants_by_template[template_id].add(variant_type)

        merged = {
            **puzzle,
            "gold_label": template["ground_truth"],
            "query_type": template["query_type"],
            "category": template["category"],
            "axioms": template["axioms"],
            "query": template["query"],
        }

        out.append(merged)

    for template_id in sorted(templates):
        observed = variants_by_template[template_id]

        if set(EXPECTED_VARIANTS) != observed:
            raise ValueError(
                f"Template {template_id} has variants {sorted(observed)}, "
                f"expected {list(EXPECTED_VARIANTS)}."
            )

    return out


if __name__ == "__main__":
    puzzles = load_puzzles()

    print(f"Loaded {len(puzzles)} puzzles")

    for puzzle in puzzles:
        print(
            f"  {puzzle['id']:24s} | "
            f"gold={puzzle['gold_label']:8s} | "
            f"type={puzzle['query_type']:14s} | "
            f"{puzzle['domain_label']}"
        )