"""Prompt construction and answer parsing.

Multiple-choice format: the four letters A/B/C/D map to the four labels
True/False/Unknown/Paradox. The mapping is determined per call by an
`option_order`, a permutation of LABELS. The canonical mapping
(A=True, B=False, C=Unknown, D=Paradox) is used when no permutation is given.

To control for positional bias the runner cycles deterministically through all
24 permutations of the four labels; see `option_order_for`.

Canonical label order (used as the default and as the index basis for systematic
counterbalancing):
    True, False, Unknown, Paradox  <->  A, B, C, D
"""

import itertools
import re

LETTER_LABEL = {
    "A": "True",
    "B": "False",
    "C": "Unknown",
    "D": "Paradox",
}

LABEL_LETTER = {label: letter for letter, label in LETTER_LABEL.items()}

LABELS = ("True", "False", "Unknown", "Paradox")
LETTER_OPTIONS = ("A", "B", "C", "D")

PERMUTATIONS: tuple[tuple[str, ...], ...] = tuple(itertools.permutations(LABELS))

_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def option_order_for(
    puzzle_idx: int, seed: int, n_seeds: int
) -> tuple[str, ...]:
    """Return the systematic counterbalancing permutation for one trial.

    Deterministically indexes into the 24 permutations of LABELS by
    `(puzzle_idx * n_seeds + seed) % 24`. With 36 puzzles x 10 seeds this gives
    exactly uniform per-permutation coverage (15 hits each).

    Pure function; reproducible across processes (no PYTHONHASHSEED dependency).
    """
    if puzzle_idx < 0:
        raise ValueError(f"puzzle_idx must be >= 0, got {puzzle_idx}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if n_seeds <= 0:
        raise ValueError(f"n_seeds must be > 0, got {n_seeds}")

    return PERMUTATIONS[(puzzle_idx * n_seeds + seed) % len(PERMUTATIONS)]


def letter_to_label_map(option_order: tuple[str, ...]) -> dict[str, str]:
    """Return the per-call mapping from option letter (A/B/C/D) to label."""
    _validate_option_order(option_order)
    return {letter: label for letter, label in zip(LETTER_OPTIONS, option_order)}


def label_to_letter_map(option_order: tuple[str, ...]) -> dict[str, str]:
    """Return the per-call mapping from label to option letter (A/B/C/D)."""
    _validate_option_order(option_order)
    return {label: letter for letter, label in zip(LETTER_OPTIONS, option_order)}


def _validate_option_order(option_order: tuple[str, ...]) -> None:
    if tuple(sorted(option_order)) != tuple(sorted(LABELS)):
        raise ValueError(
            f"option_order must be a permutation of {LABELS}, got {option_order!r}"
        )


def build_prompt(
    puzzle: dict, option_order: tuple[str, ...] | None = None
) -> str:
    """Build the prompt shown to the model.

    The prompt ends with 'Answer:' and asks the model to output exactly one
    option letter. If `option_order` is given, the four labels are listed in
    that order under A/B/C/D; otherwise the canonical LABELS order is used.
    """
    if option_order is None:
        option_order = LABELS
    else:
        _validate_option_order(option_order)

    return (
        f"{puzzle['narrative']}\n\n"
        f"{puzzle['question']}\n\n"
        f"Choose exactly one of the following options:\n"
        f"A. {option_order[0]}\n"
        f"B. {option_order[1]}\n"
        f"C. {option_order[2]}\n"
        f"D. {option_order[3]}\n\n"
        f"Your answer must be exactly one letter: A, B, C, or D.\n"
        f"Do not write anything else.\n"
        f"Answer:"
    )


def parse_letter_token(raw: str) -> str | None:
    """Parse a generated model answer into one of A/B/C/D.

    Returns the uppercase option letter, or None if no valid standalone option
    letter can be found.
    """
    if not raw:
        return None

    match = _LETTER_RE.search(raw.strip())

    if match is None:
        return None

    return match.group(1).upper()


def parse_letter(
    raw: str, option_order: tuple[str, ...] | None = None
) -> str | None:
    """Parse a generated answer into one of the four labels.

    Returns one of True, False, Unknown, Paradox, or None if no valid standalone
    option letter can be found. If `option_order` is given, the letter is
    mapped to a label via that permutation; otherwise the canonical
    LETTER_LABEL mapping is used.

    Accepted examples:
        A
        a
        A.
        (A)
        Answer: A

    Rejected examples:
        Because
        Cat
        Unknown
    """
    letter = parse_letter_token(raw)

    if letter is None:
        return None

    if option_order is None:
        return LETTER_LABEL[letter]

    return letter_to_label_map(option_order)[letter]


def label_to_letter(label: str) -> str:
    """Return the canonical option letter corresponding to a label."""
    if label not in LABEL_LETTER:
        raise ValueError(f"Unknown label: {label!r}")

    return LABEL_LETTER[label]


def letter_to_label(letter: str) -> str:
    """Return the canonical label corresponding to an option letter."""
    normalized = letter.strip().upper()

    if normalized not in LETTER_LABEL:
        raise ValueError(f"Unknown option letter: {letter!r}")

    return LETTER_LABEL[normalized]
