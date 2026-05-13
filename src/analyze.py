"""Analyze a results CSV produced by run.py.

Handles repeated sampled answer selection:
  - one row per (puzzle, seed)
  - option_order = the permutation of LABELS assigned to (A, B, C, D) for that trial
  - sampled_letter / sampled_label = letter sampled by the model and the label it maps to
  - argmax_letter / argmax_label = argmax letter and the label it maps to
  - is_invalid = sanity-check flag for parser/wrapper mismatch

Reports:
  - invalid rate
  - per-run sampled accuracy
  - per-run argmax accuracy
  - positional diagnostics (chosen-letter distribution, accuracy by gold-letter position)
  - accuracy by variant/template/category
  - majority-vote accuracy per puzzle
  - response stability across seeds
  - cross-variant consistency on majority predictions
  - confidence and entropy diagnostics
  - cluster bootstrap CIs over puzzle_id
  - paired bootstrap deltas over template_id for variant comparisons

Usage:
    python analyze.py path/to/results_qwen3b_TIMESTAMP.csv
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ("True", "False", "Unknown", "Paradox")
LETTER_OPTIONS = ("A", "B", "C", "D")
VARIANTS = ("familiar", "belief_violating", "artificial", "abstract")


def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a results CSV and coerce numeric/boolean fields."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["seed"] = int(row["seed"])
        row["temperature"] = float(row["temperature"])
        row["top_p"] = float(row["top_p"])
        row["max_prob"] = float(row["max_prob"])
        row["entropy_nats"] = float(row["entropy_nats"])

        row["is_invalid"] = row["is_invalid"].strip().lower() in (
            "true",
            "1",
            "yes",
        )
        row["correct_sampled"] = row["correct_sampled"].strip().lower() in (
            "true",
            "1",
            "yes",
        )
        row["correct_argmax"] = row["correct_argmax"].strip().lower() in (
            "true",
            "1",
            "yes",
        )

        if "option_order" in row and row["option_order"]:
            row["option_order"] = tuple(row["option_order"].split("|"))

        for label in LABELS:
            row[f"prob_{label}"] = float(row[f"prob_{label}"])
            row[f"logprob_{label}"] = float(row[f"logprob_{label}"])

        for letter in LETTER_OPTIONS:
            if f"prob_{letter}" in row:
                row[f"prob_{letter}"] = float(row[f"prob_{letter}"])
                row[f"logprob_{letter}"] = float(row[f"logprob_{letter}"])

    return rows


def gold_letter(row: dict[str, Any]) -> str | None:
    """Return the option letter (A/B/C/D) at which the gold label sits.

    Requires the row to have an `option_order` tuple. Returns None if absent.
    """
    option_order = row.get("option_order")

    if not option_order:
        return None

    try:
        return LETTER_OPTIONS[option_order.index(row["gold_label"])]
    except ValueError:
        return None


def fmt_acc(correct: int, total: int) -> str:
    """Format accuracy as count and percentage."""
    return f"{correct}/{total} = {correct / total:.1%}" if total else "0/0"


def mean(values: list[float]) -> float | None:
    """Return the mean of a list, or None if empty."""
    return sum(values) / len(values) if values else None


def modal_label(labels: list[str]) -> tuple[str | None, int, bool]:
    """Return modal label, modal count, and whether there is a tie.

    Ties are broken deterministically using LABELS order.
    """
    if not labels:
        return None, 0, False

    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    has_tie = len(tied) > 1

    for label in LABELS:
        if label in tied:
            return label, max_count, has_tie

    return tied[0], max_count, has_tie


def cluster_bootstrap_accuracy(
    rows: list[dict[str, Any]],
    correct_key: str,
    cluster_key: str = "puzzle_id",
    n_iter: int = 2000,
    conf: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster bootstrap accuracy by resampling clusters, not rows.

    Each bootstrap sample draws cluster IDs with replacement and includes all
    rows belonging to the sampled clusters.
    """
    rng = random.Random(rng_seed)

    if not rows:
        return 0.0, 0.0, 0.0

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        clusters[row[cluster_key]].append(row)

    cluster_ids = list(clusters)

    if not cluster_ids:
        return 0.0, 0.0, 0.0

    observed = sum(float(row[correct_key]) for row in rows) / len(rows)

    boot_means = []

    for _ in range(n_iter):
        sampled_ids = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_rows = []

        for cluster_id in sampled_ids:
            sampled_rows.extend(clusters[cluster_id])

        acc = sum(float(row[correct_key]) for row in sampled_rows) / len(
            sampled_rows
        )
        boot_means.append(acc)

    boot_means.sort()

    lo_idx = int((1 - conf) / 2 * n_iter)
    hi_idx = int((1 + conf) / 2 * n_iter)
    hi_idx = min(hi_idx, n_iter - 1)

    return observed, boot_means[lo_idx], boot_means[hi_idx]


def cluster_bootstrap_mean(
    rows: list[dict[str, Any]],
    value_key: str,
    cluster_key: str = "puzzle_id",
    n_iter: int = 2000,
    conf: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster bootstrap mean by resampling clusters, not rows."""
    rng = random.Random(rng_seed)

    if not rows:
        return 0.0, 0.0, 0.0

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        clusters[row[cluster_key]].append(row)

    cluster_ids = list(clusters)

    if not cluster_ids:
        return 0.0, 0.0, 0.0

    observed = sum(float(row[value_key]) for row in rows) / len(rows)

    boot_means = []

    for _ in range(n_iter):
        sampled_ids = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_values = []

        for cluster_id in sampled_ids:
            sampled_values.extend(float(row[value_key]) for row in clusters[cluster_id])

        boot_means.append(sum(sampled_values) / len(sampled_values))

    boot_means.sort()

    lo_idx = int((1 - conf) / 2 * n_iter)
    hi_idx = int((1 + conf) / 2 * n_iter)
    hi_idx = min(hi_idx, n_iter - 1)

    return observed, boot_means[lo_idx], boot_means[hi_idx]


def build_puzzle_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated runs into one summary row per puzzle."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["puzzle_id"]].append(row)

    summaries = []

    for puzzle_id, puzzle_rows in sorted(grouped.items()):
        valid_rows = [row for row in puzzle_rows if not row["is_invalid"]]
        labels = [row["sampled_label"] for row in valid_rows]

        mode, modal_count, has_tie = modal_label(labels)

        gold = puzzle_rows[0]["gold_label"]
        stability = modal_count / len(labels) if labels else 0.0
        majority_correct = mode == gold if mode is not None else False

        summaries.append(
            {
                "puzzle_id": puzzle_id,
                "template_id": puzzle_rows[0]["template_id"],
                "variant_type": puzzle_rows[0]["variant_type"],
                "category": puzzle_rows[0]["category"],
                "domain_label": puzzle_rows[0]["domain_label"],
                "gold_label": gold,
                "majority_label": mode,
                "majority_correct": majority_correct,
                "stability": stability,
                "has_tie": has_tie,
                "n_runs": len(puzzle_rows),
                "n_valid": len(valid_rows),
                "n_invalid": len(puzzle_rows) - len(valid_rows),
                "vote_counts": Counter(labels),
                "mean_confidence": mean([row["max_prob"] for row in puzzle_rows]),
                "mean_entropy": mean([row["entropy_nats"] for row in puzzle_rows]),
            }
        )

    return summaries


def paired_variant_delta_bootstrap(
    summaries: list[dict[str, Any]],
    variant_a: str,
    variant_b: str,
    metric_key: str = "majority_correct",
    n_iter: int = 2000,
    conf: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap over template_id for variant comparisons.

    Uses one summary row per puzzle. For each template, compares variant_a and
    variant_b, then resamples templates with replacement.
    """
    rng = random.Random(rng_seed)

    by_template: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in summaries:
        by_template[row["template_id"]][row["variant_type"]] = row

    paired_templates = [
        template_id
        for template_id, variant_map in by_template.items()
        if variant_a in variant_map and variant_b in variant_map
    ]

    if not paired_templates:
        return 0.0, 0.0, 0.0

    def compute_delta(template_ids: list[str]) -> float:
        a_values = []
        b_values = []

        for template_id in template_ids:
            variant_map = by_template[template_id]
            a_values.append(float(variant_map[variant_a][metric_key]))
            b_values.append(float(variant_map[variant_b][metric_key]))

        return sum(a_values) / len(a_values) - sum(b_values) / len(b_values)

    observed = compute_delta(paired_templates)

    boot_deltas = []

    for _ in range(n_iter):
        sampled_templates = [
            rng.choice(paired_templates) for _ in paired_templates
        ]
        boot_deltas.append(compute_delta(sampled_templates))

    boot_deltas.sort()

    lo_idx = int((1 - conf) / 2 * n_iter)
    hi_idx = int((1 + conf) / 2 * n_iter)
    hi_idx = min(hi_idx, n_iter - 1)

    return observed, boot_deltas[lo_idx], boot_deltas[hi_idx]


def print_group_accuracy_with_cluster_ci(
    rows: list[dict[str, Any]],
    group_key: str,
    correct_key: str,
    cluster_key: str,
    title: str,
    ordered_keys: tuple[str, ...] | None = None,
    n_iter: int = 2000,
    rng_seed: int = 42,
) -> None:
    """Print grouped accuracy with cluster-bootstrap CI."""
    print(title)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row[group_key]].append(row)

    keys = list(ordered_keys) if ordered_keys else sorted(grouped)

    for key in keys:
        group_rows = grouped.get(key, [])

        if not group_rows:
            print(f"  {key:20s} 0/0")
            continue

        correct = sum(int(row[correct_key]) for row in group_rows)
        total = len(group_rows)
        m, lo, hi = cluster_bootstrap_accuracy(
            group_rows,
            correct_key=correct_key,
            cluster_key=cluster_key,
            n_iter=n_iter,
            rng_seed=rng_seed,
        )

        print(
            f"  {key:20s} {fmt_acc(correct, total)}   "
            f"cluster CI: {m:.3f} [{lo:.3f}, {hi:.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="path to a results CSV from run.py")
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="number of bootstrap iterations",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap procedures",
    )

    args = parser.parse_args()

    rows = load_csv(args.csv)

    n_rows = len(rows)
    n_puzzles = len({row["puzzle_id"] for row in rows})
    n_templates = len({row["template_id"] for row in rows})
    n_seeds = len({row["seed"] for row in rows})

    print(f"Loaded {n_rows} rows from {args.csv.name}")
    print(f"  {n_puzzles} puzzles × {n_seeds} seeds")
    print(f"  {n_templates} templates")

    model_names = sorted({row["model_name"] for row in rows if row.get("model_name")})
    if len(model_names) == 1:
        print(f"  Model: {model_names[0]}")
    elif len(model_names) > 1:
        print(f"  Models ({len(model_names)} distinct):")
        for name in model_names:
            print(f"    - {name}")

    print()

    if n_puzzles != 36:
        print(f"Warning: expected 36 puzzles, found {n_puzzles}")

    if n_templates != 9:
        print(f"Warning: expected 9 templates, found {n_templates}")

    expected_rows = n_puzzles * n_seeds

    if n_rows != expected_rows:
        print(f"Warning: expected {expected_rows} rows, found {n_rows}")

    unexpected_variants = sorted(
        {row["variant_type"] for row in rows} - set(VARIANTS)
    )

    if unexpected_variants:
        print("Warning: unexpected variant_type values found:")
        for variant in unexpected_variants:
            print(f"  {variant}")
        print()

    # 1. Invalid-rate sanity check
    invalid = sum(row["is_invalid"] for row in rows)

    print("=== Sanity check ===")
    print(f"  Invalid outputs: {invalid}/{n_rows} = {invalid / n_rows:.1%}")

    if invalid:
        print("  WARNING: invalid outputs should be 0 under constrained sampling.")

    print()

    # 2. Per-run accuracy
    sampled_correct = sum(row["correct_sampled"] for row in rows)
    argmax_correct = sum(row["correct_argmax"] for row in rows)

    sampled_m, sampled_lo, sampled_hi = cluster_bootstrap_accuracy(
        rows,
        correct_key="correct_sampled",
        cluster_key="puzzle_id",
        n_iter=args.bootstrap_iters,
        rng_seed=args.bootstrap_seed,
    )

    argmax_m, argmax_lo, argmax_hi = cluster_bootstrap_accuracy(
        rows,
        correct_key="correct_argmax",
        cluster_key="puzzle_id",
        n_iter=args.bootstrap_iters,
        rng_seed=args.bootstrap_seed,
    )

    print("=== Per-run accuracy ===")
    print(
        f"  Sampled label: {fmt_acc(sampled_correct, n_rows)}   "
        f"cluster CI: {sampled_m:.3f} [{sampled_lo:.3f}, {sampled_hi:.3f}]"
    )
    print(
        f"  Argmax label:  {fmt_acc(argmax_correct, n_rows)}   "
        f"cluster CI: {argmax_m:.3f} [{argmax_lo:.3f}, {argmax_hi:.3f}]"
    )
    print()

    # 2b. Positional diagnostics (only meaningful under counterbalancing).
    has_option_order = all(isinstance(row.get("option_order"), tuple) for row in rows)
    has_sampled_letter = all(row.get("sampled_letter") for row in rows)

    if has_sampled_letter:
        print("=== Positional diagnostics ===")

        print("Chosen-letter distribution (diagnostic for residual positional bias):")
        chosen_counts = Counter(row["sampled_letter"] for row in rows)
        for letter in LETTER_OPTIONS:
            count = chosen_counts.get(letter, 0)
            print(f"  {letter}: {count}/{n_rows} = {count / n_rows:.1%}")

        has_letter_probs = all(f"prob_{LETTER_OPTIONS[0]}" in row for row in rows)

        if has_letter_probs:
            print()
            print("Mean letter-level probability (positional preference, "
                  "independent of label assignment):")
            for letter in LETTER_OPTIONS:
                m = mean([row[f"prob_{letter}"] for row in rows])
                if m is not None:
                    print(f"  prob_{letter}: {m:.3f}")

        print()
        print("Sampled accuracy by chosen letter:")
        for letter in LETTER_OPTIONS:
            letter_rows = [row for row in rows if row["sampled_letter"] == letter]
            correct = sum(int(row["correct_sampled"]) for row in letter_rows)
            total = len(letter_rows)
            print(f"  {letter}: {fmt_acc(correct, total)}")

        if has_option_order:
            print()
            print("Sampled accuracy by gold-label position (where the correct option"
                  " was placed):")
            position_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                gl = gold_letter(row)
                if gl is not None:
                    position_groups[gl].append(row)

            for letter in LETTER_OPTIONS:
                group = position_groups.get(letter, [])
                correct = sum(int(row["correct_sampled"]) for row in group)
                total = len(group)
                print(f"  gold@{letter}: {fmt_acc(correct, total)}")

            print()
            n_perms = len({row["option_order"] for row in rows})
            print(
                f"Distinct option_order permutations observed: {n_perms} (of 24 possible)"
            )

        print()

    # 3. Accuracy by variant/template/category
    print_group_accuracy_with_cluster_ci(
        rows,
        group_key="variant_type",
        correct_key="correct_sampled",
        cluster_key="puzzle_id",
        title="Sampled accuracy by variant_type:",
        ordered_keys=VARIANTS,
        n_iter=args.bootstrap_iters,
        rng_seed=args.bootstrap_seed,
    )

    print()

    print_group_accuracy_with_cluster_ci(
        rows,
        group_key="template_id",
        correct_key="correct_sampled",
        cluster_key="puzzle_id",
        title="Sampled accuracy by template_id:",
        n_iter=args.bootstrap_iters,
        rng_seed=args.bootstrap_seed,
    )

    print()

    print_group_accuracy_with_cluster_ci(
        rows,
        group_key="category",
        correct_key="correct_sampled",
        cluster_key="puzzle_id",
        title="Sampled accuracy by category:",
        n_iter=args.bootstrap_iters,
        rng_seed=args.bootstrap_seed,
    )

    # 4. Majority-vote and stability
    summaries = build_puzzle_summary(rows)

    majority_correct = sum(summary["majority_correct"] for summary in summaries)
    mean_stability = mean([summary["stability"] for summary in summaries])
    tied_majorities = sum(summary["has_tie"] for summary in summaries)

    print("\n=== Majority-vote analysis per puzzle ===")
    print(f"  Majority-vote accuracy: {fmt_acc(majority_correct, len(summaries))}")

    if mean_stability is not None:
        print(f"  Mean response stability: {mean_stability:.3f}")

    print(f"  Tied majorities: {tied_majorities}/{len(summaries)} puzzles")

    # Majority-vote accuracy by variant_type
    print("\nMajority-vote accuracy by variant_type:")
    summaries_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for summary in summaries:
        summaries_by_variant[summary["variant_type"]].append(summary)

    for variant in VARIANTS:
        variant_summaries = summaries_by_variant[variant]
        correct = sum(summary["majority_correct"] for summary in variant_summaries)
        total = len(variant_summaries)

        print(f"  {variant:18s} {fmt_acc(correct, total)}")

    # Stability by variant_type
    print("\nMean stability by variant_type:")
    for variant in VARIANTS:
        values = [summary["stability"] for summary in summaries_by_variant[variant]]
        m = mean(values)

        if m is not None:
            print(f"  {variant:18s} {m:.3f}")

    # 5. Cross-variant consistency on majority predictions
    print("\nCross-variant consistency on majority predictions:")

    template_to_variant_majority: dict[str, dict[str, str | None]] = defaultdict(dict)

    for summary in summaries:
        template_to_variant_majority[summary["template_id"]][
            summary["variant_type"]
        ] = summary["majority_label"]

    consistent = 0
    incomplete = []

    for template_id, variant_map in template_to_variant_majority.items():
        if len(variant_map) != len(VARIANTS):
            incomplete.append(template_id)

        if len(variant_map) == len(VARIANTS) and len(set(variant_map.values())) == 1:
            consistent += 1

    print(
        f"  {consistent}/{len(template_to_variant_majority)} templates "
        f"have identical majority prediction across all 4 variants"
    )

    inconsistent = {
        template_id: variant_map
        for template_id, variant_map in sorted(template_to_variant_majority.items())
        if len(set(variant_map.values())) > 1
    }

    if inconsistent:
        print("  inconsistent templates:")
        for template_id, variant_map in inconsistent.items():
            ordered = {
                variant: variant_map.get(variant, "<missing>")
                for variant in VARIANTS
            }
            print(f"    {template_id}: {ordered}")

    if incomplete:
        print("  warning: incomplete templates:")
        for template_id in incomplete:
            print(f"    {template_id}")

    # 6. Paired bootstrap variant deltas
    print("\nPaired bootstrap deltas on majority-vote accuracy:")
    comparisons = (
        ("familiar", "belief_violating"),
        ("artificial", "belief_violating"),
        ("abstract", "belief_violating"),
        ("abstract", "familiar"),
    )

    for variant_a, variant_b in comparisons:
        delta, lo, hi = paired_variant_delta_bootstrap(
            summaries,
            variant_a=variant_a,
            variant_b=variant_b,
            metric_key="majority_correct",
            n_iter=args.bootstrap_iters,
            rng_seed=args.bootstrap_seed,
        )

        print(
            f"  {variant_a:18s} - {variant_b:18s}: "
            f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        )

    # 7. Confidence and entropy diagnostics
    print("\nMean confidence (max_prob) by sampled correctness:")

    correct_conf = [row["max_prob"] for row in rows if row["correct_sampled"]]
    wrong_conf = [
        row["max_prob"]
        for row in rows
        if not row["correct_sampled"] and not row["is_invalid"]
    ]
    invalid_conf = [row["max_prob"] for row in rows if row["is_invalid"]]

    m = mean(correct_conf)
    if m is not None:
        print(f"  Correct  (n={len(correct_conf):4d}): mean = {m:.3f}")

    m = mean(wrong_conf)
    if m is not None:
        print(f"  Wrong    (n={len(wrong_conf):4d}): mean = {m:.3f}")

    m = mean(invalid_conf)
    if m is not None:
        print(f"  Invalid  (n={len(invalid_conf):4d}): mean = {m:.3f}")

    print("\nMean entropy (nats) by sampled correctness:")

    correct_entropy = [row["entropy_nats"] for row in rows if row["correct_sampled"]]
    wrong_entropy = [
        row["entropy_nats"]
        for row in rows
        if not row["correct_sampled"] and not row["is_invalid"]
    ]
    invalid_entropy = [row["entropy_nats"] for row in rows if row["is_invalid"]]

    m = mean(correct_entropy)
    if m is not None:
        print(f"  Correct  (n={len(correct_entropy):4d}): mean H = {m:.3f}")

    m = mean(wrong_entropy)
    if m is not None:
        print(f"  Wrong    (n={len(wrong_entropy):4d}): mean H = {m:.3f}")

    m = mean(invalid_entropy)
    if m is not None:
        print(f"  Invalid  (n={len(invalid_entropy):4d}): mean H = {m:.3f}")

    print("\nMean confidence and entropy by variant_type:")

    conf_by_variant: dict[str, list[float]] = defaultdict(list)
    entropy_by_variant: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        conf_by_variant[row["variant_type"]].append(row["max_prob"])
        entropy_by_variant[row["variant_type"]].append(row["entropy_nats"])

    for variant in VARIANTS:
        conf_mean = mean(conf_by_variant[variant])
        ent_mean = mean(entropy_by_variant[variant])

        if conf_mean is not None and ent_mean is not None:
            print(
                f"  {variant:18s} "
                f"mean conf = {conf_mean:.3f}   mean H = {ent_mean:.3f}"
            )

    # 8. Per-puzzle vote distributions
    print("\nPer-puzzle majority predictions and stability:")
    for summary in summaries:
        counts = dict(summary["vote_counts"])

        print(
            f"  {summary['puzzle_id']:24s} "
            f"gold={summary['gold_label']:8s} "
            f"majority={str(summary['majority_label']):8s} "
            f"correct={str(summary['majority_correct']):5s} "
            f"stability={summary['stability']:.2f} "
            f"invalid={summary['n_invalid']}/{summary['n_runs']} "
            f"votes={counts}"
        )


if __name__ == "__main__":
    main()