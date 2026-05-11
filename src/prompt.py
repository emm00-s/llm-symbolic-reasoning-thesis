"""Prompt construction and answer parsing.

Multiple-choice format: A/B/C/D map to True/False/Unknown/Paradox respectively.

The prompt instructs the model to answer with exactly one option letter. Since
instruction-tuned models may still produce minor formatting variants, generated
outputs are parsed with a robust regex that accepts standalone A/B/C/D letters
regardless of casing or surrounding punctuation.

Canonical label order used everywhere in the pipeline:
    True, False, Unknown, Paradox  <->  A, B, C, D
"""

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

_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def build_prompt(puzzle: dict) -> str:
    """Build the prompt shown to the model.

    The prompt ends with 'Answer:' and asks the model to output exactly one
    option letter.
    """
    return (
        f"{puzzle['narrative']}\n\n"
        f"{puzzle['question']}\n\n"
        f"Choose exactly one of the following options:\n"
        f"A. True\n"
        f"B. False\n"
        f"C. Unknown\n"
        f"D. Paradox\n\n"
        f"Your answer must be exactly one letter: A, B, C, or D.\n"
        f"Do not write anything else.\n"
        f"Answer:"
    )


def parse_letter(raw: str) -> str | None:
    """Parse a generated model answer into one of the four labels.

    Returns one of True, False, Unknown, Paradox, or None if no valid standalone
    option letter can be found.

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
    if not raw:
        return None

    match = _LETTER_RE.search(raw.strip())

    if match is None:
        return None

    letter = match.group(1).upper()

    return LETTER_LABEL[letter]


def label_to_letter(label: str) -> str:
    """Return the option letter corresponding to a label."""
    if label not in LABEL_LETTER:
        raise ValueError(f"Unknown label: {label!r}")

    return LABEL_LETTER[label]


def letter_to_label(letter: str) -> str:
    """Return the label corresponding to an option letter."""
    normalized = letter.strip().upper()

    if normalized not in LETTER_LABEL:
        raise ValueError(f"Unknown option letter: {letter!r}")

    return LETTER_LABEL[normalized]