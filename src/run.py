"""Run the 36-puzzle benchmark with repeated sampled answer selection.

For each puzzle and each seed, the model samples one answer among A/B/C/D and
the script saves:

  - model_name: Hugging Face repo id of the model evaluated
  - option_order: per-trial permutation of (True, False, Unknown, Paradox)
    assigned to (A, B, C, D), drawn from a systematic counterbalancing
    schedule across all 24 permutations
  - raw_output: canonical generated answer letter
  - sampled_letter: option letter actually sampled (A/B/C/D)
  - sampled_label: label that `sampled_letter` maps to under `option_order`
  - argmax_letter: most probable option letter according to first-token logprobs
  - argmax_label: label that `argmax_letter` maps to under `option_order`
  - parsed_label: label recovered from raw_output by the parser
  - is_invalid: sanity check flag for parser/wrapper mismatch
  - correct_sampled: sampled_label == gold_label
  - correct_argmax: argmax_label == gold_label
  - first-token logprobs and normalized probabilities over the 4 labels
  - first-token logprobs and normalized probabilities over the 4 letters
    A/B/C/D (positional-bias diagnostic, independent of option_order)
  - max_prob (over labels) and entropy

CSV has one row per (puzzle, seed). Aggregate with analyze.py.
"""

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

from .dataset import load_puzzles
from .model import DEFAULT_MODEL_NAME
from .prompt import (
    LABELS,
    LETTER_OPTIONS,
    build_prompt,
    option_order_for,
    parse_letter,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"


def entropy_nats(probs: dict[str, float]) -> float:
    """Shannon entropy in nats over a probability distribution."""
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


def run_one(
    puzzle: dict,
    seed: int,
    temperature: float,
    top_p: float,
    option_order: tuple[str, ...],
    model_name: str,
) -> dict:
    """Run one sampled answer selection for one puzzle and return a CSV row."""
    from .model import call_llm

    prompt = build_prompt(puzzle, option_order=option_order)

    out = call_llm(
        prompt=prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        option_order=option_order,
        model_name=model_name,
    )

    sampled_label = out["sampled_label"]
    argmax_label = out["argmax_label"]
    parsed_label = parse_letter(out["raw_text"], option_order=option_order)

    # Sanity check: under constrained answer selection, this should always be False.
    # If True, there is a mismatch between model output, tokenizer mapping, or parser.
    is_invalid = parsed_label is None or parsed_label != sampled_label

    logprobs = out["first_token_logprobs"]
    probs = out["first_token_probs"]

    correct_sampled = sampled_label == puzzle["gold_label"] and not is_invalid
    correct_argmax = argmax_label == puzzle["gold_label"]

    row = {
        "puzzle_id": puzzle["id"],
        "template_id": puzzle["template_id"],
        "variant_type": puzzle["variant_type"],
        "category": puzzle["category"],
        "domain_label": puzzle["domain_label"],
        "gold_label": puzzle["gold_label"],
        "model_name": model_name,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "option_order": "|".join(option_order),
        "raw_output": out["raw_text"],
        "sampled_letter": out["sampled_letter"],
        "sampled_label": sampled_label,
        "argmax_letter": out["argmax_letter"],
        "argmax_label": argmax_label,
        "parsed_label": parsed_label if parsed_label else "",
        "is_invalid": is_invalid,
        "correct_sampled": correct_sampled,
        "correct_argmax": correct_argmax,
        "max_prob": max(probs.values()),
        "entropy_nats": entropy_nats(probs),
    }

    for label in LABELS:
        row[f"logprob_{label}"] = logprobs[label]
        row[f"prob_{label}"] = probs[label]

    letter_logprobs = out["first_token_letter_logprobs"]
    letter_probs = out["first_token_letter_probs"]

    for letter in LETTER_OPTIONS:
        row[f"logprob_{letter}"] = letter_logprobs[letter]
        row[f"prob_{letter}"] = letter_probs[letter]

    return row


def save_csv(rows: list[dict], path: Path) -> None:
    """Save result rows to CSV."""
    if not rows:
        raise ValueError("No rows to save.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark runner with repeated sampled answer selection"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of seeds per puzzle",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        dest="top_p",
        help=(
            "top-p value applied over the four-label distribution. "
            "Default 1.0 (no truncation): with K=4 labels there is no vocabulary "
            "tail, and lower-probability options (Unknown / Paradox) often carry "
            "meaningful signal."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        dest="model_name",
        help=(
            "Hugging Face causal-LM repo id to evaluate "
            f"(default: {DEFAULT_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--tag",
        default="qwen3b",
        help="filename tag for output CSV",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-run output",
    )

    args = parser.parse_args()

    if args.seeds <= 0:
        raise ValueError("--seeds must be greater than 0.")

    if args.temperature < 0:
        raise ValueError("--temperature must be greater than or equal to 0.")

    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")

    puzzles = load_puzzles()
    # Stable puzzle index for systematic counterbalancing: sort by puzzle id so
    # the index is independent of JSON-file ordering.
    puzzles = sorted(puzzles, key=lambda p: p["id"])
    total = len(puzzles) * args.seeds

    print(f"Loaded {len(puzzles)} puzzles × {args.seeds} seeds = {total} calls.")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}, top_p: {args.top_p}\n")

    rows = []
    counter = 0

    for puzzle_idx, puzzle in enumerate(puzzles):
        for seed in range(args.seeds):
            counter += 1

            option_order = option_order_for(
                puzzle_idx=puzzle_idx,
                seed=seed,
                n_seeds=args.seeds,
            )

            row = run_one(
                puzzle=puzzle,
                seed=seed,
                temperature=args.temperature,
                top_p=args.top_p,
                option_order=option_order,
                model_name=args.model_name,
            )

            rows.append(row)

            if not args.quiet:
                if row["is_invalid"]:
                    mark = "INV"
                elif row["correct_sampled"]:
                    mark = "OK"
                else:
                    mark = "FAIL"

                argmax_mark = "OK" if row["correct_argmax"] else "FAIL"
                preview = row["raw_output"][:20].replace("\n", "\\n")

                print(
                    f"  [{counter:4d}/{total}] "
                    f"[sample={mark:4s}] "
                    f"[argmax={argmax_mark:4s}] "
                    f"{row['puzzle_id']:24s} "
                    f"seed={seed:2d} "
                    f"gold={row['gold_label']:8s} "
                    f"sampled={row['sampled_label']:8s}({row['sampled_letter']}) "
                    f"argmax={row['argmax_label']:8s}({row['argmax_letter']}) "
                    f"conf={row['max_prob']:.2f} "
                    f"H={row['entropy_nats']:.2f} "
                    f"raw={preview!r}"
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = (
        OUTPUT_DIR
        / f"results_{args.tag}_T{args.temperature}_topP{args.top_p}_n{args.seeds}_{timestamp}.csv"
    )

    save_csv(rows, out_path)


if __name__ == "__main__":
    main()
