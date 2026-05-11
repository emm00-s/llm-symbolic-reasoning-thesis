"""Z3 verification for the 9 formal templates.

Each template's z3_code field in templates.json is a self-contained Python
snippet that imports z3, builds the axioms over X1..Xn, and assigns the computed
gold label to a variable named result.

The z3_code field is treated as trusted hand-authored code distributed with the
benchmark. Do not load templates.json from an untrusted source, because this
would amount to executing arbitrary Python code.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PUZZLES_DIR = Path(__file__).resolve().parent.parent / "puzzles"

VALID_LABELS = {"True", "False", "Unknown", "Paradox"}
EXPECTED_NUM_TEMPLATES = 9

REQUIRED_TEMPLATE_FIELDS = {
    "template_id",
    "ground_truth",
    "query_type",
    "category",
    "axioms",
    "query",
    "z3_code",
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
        identifier = item.get("template_id") or "<unknown>"
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


@lru_cache(maxsize=1)
def _load_templates(puzzles_dir: Path = PUZZLES_DIR) -> dict[str, dict[str, Any]]:
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

        declared = template["ground_truth"]

        if declared not in VALID_LABELS:
            raise ValueError(
                f"Template {template['template_id']!r} has invalid ground_truth "
                f"{declared!r}. Expected one of {sorted(VALID_LABELS)}."
            )

    _check_unique(templates_data, "template_id", "templates.json")

    return {template["template_id"]: template for template in templates_data}


def verify(template_id: str) -> str:
    """Execute a template's z3_code and return its computed gold label.

    Returns one of: True, False, Unknown, Paradox.
    """
    templates = _load_templates()

    if template_id not in templates:
        raise KeyError(f"Unknown template_id: {template_id!r}")

    namespace: dict[str, Any] = {}

    exec(templates[template_id]["z3_code"], namespace)

    if "result" not in namespace:
        raise RuntimeError(
            f"z3_code for {template_id} did not assign a variable named result."
        )

    result = namespace["result"]

    if result not in VALID_LABELS:
        raise ValueError(
            f"z3_code for {template_id} returned invalid label {result!r}. "
            f"Expected one of {sorted(VALID_LABELS)}."
        )

    return result


def verify_all() -> dict[str, str]:
    """Run verify() on all templates and return {template_id: gold_label}."""
    templates = _load_templates()

    return {template_id: verify(template_id) for template_id in sorted(templates)}


def cross_check_ground_truth() -> list[str]:
    """Check that each declared ground_truth matches its z3_code result.

    Returns a list of mismatch messages. The list is empty if all templates are
    consistent.
    """
    templates = _load_templates()
    errors: list[str] = []

    for template_id, template in sorted(templates.items()):
        actual = verify(template_id)
        declared = template["ground_truth"]

        if actual != declared:
            errors.append(
                f"{template_id}: z3_code returned {actual!r}, "
                f"but declared ground_truth is {declared!r}."
            )

    return errors


if __name__ == "__main__":
    print("Running Z3 verification on all 9 templates...\n")

    results = verify_all()

    for template_id, label in results.items():
        print(f"  {template_id}: {label}")

    print("\nCross-checking declared ground_truth:")

    errors = cross_check_ground_truth()

    if errors:
        print("FAIL:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("  all 9 templates: declared ground_truth matches z3_code result")