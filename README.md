# LLM Symbolic Reasoning Thesis

Controlled diagnostic benchmark for evaluating logical invariance in language models under narrative variation.

## Project structure

- `puzzles/puzzles.json`: natural-language puzzle variants
- `puzzles/templates.json`: formal templates, gold labels, and Z3 verification code
- `src/dataset.py`: loads and joins puzzles with templates
- `src/solvers.py`: verifies gold labels with Z3
- `src/prompt.py`: builds A/B/C/D prompts under a per-trial option-letter permutation and parses answers
- `src/model.py`: wraps Qwen-2.5-3B-Instruct and samples answers from the model distribution over A/B/C/D
- `src/run.py`: runs repeated sampled evaluations under systematic counterbalancing
- `src/analyze.py`: computes accuracy, majority vote, stability, cross-variant consistency, positional diagnostics, and confidence diagnostics

## Dataset

The benchmark contains:

- 9 logical templates
- 4 variants per template
- 36 total puzzles

The four variant types are:

- `familiar`
- `belief_violating`
- `artificial`
- `abstract`

## Option-letter counterbalancing

The task uses a four-option multiple-choice format with answer letters A/B/C/D. To avoid confounding logical reasoning with positional bias, the assignment of the four labels to the four letters is **not fixed**: each (puzzle, seed) trial draws one of the 24 permutations of `(True, False, Unknown, Paradox)` from a deterministic systematic schedule, giving uniform per-permutation coverage (15 hits each at 36 puzzles x 10 seeds).

The per-trial permutation is recorded in the `option_order` column of the results CSV. Letter-level diagnostics (`prob_A`/`prob_B`/`prob_C`/`prob_D`, `logprob_A`/.../`logprob_D`) are saved alongside the label-level columns so positional preference can be inspected independently of label assignment.

## Execution mode

The `src/` directory is a Python package.

All scripts should be run from the repository root using module mode:

```bash
python -m src.dataset
python -m src.solvers
python -m src.run --seeds 10 --temperature 0.7 --top-p 1.0 --tag qwen3b
python -m src.analyze results/<RESULT_FILE>.csv
```

Do not run scripts as:

```bash
python src/run.py
```

because the project uses package-relative imports.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

- `torch`
- `transformers`
- `z3-solver`

## Validation

Before running the model, validate the dataset and formal labels:

```bash
python -m src.dataset
python -m src.solvers
```

Expected checks:

- `dataset.py` should load 36 puzzles
- `solvers.py` should confirm that all 9 declared gold labels match the Z3 results

## Running experiments

A small smoke test:

```bash
python -m src.run --seeds 2 --temperature 0.7 --top-p 1.0 --tag smoke
```

Full run:

```bash
python -m src.run --seeds 10 --temperature 0.7 --top-p 1.0 --tag qwen3b
```

Results are saved in:

```text
results/
```

## Analyzing results

After generating a CSV file:

```bash
python -m src.analyze results/<RESULT_FILE>.csv
```

The analysis reports:

- invalid-output sanity check
- sampled per-run accuracy
- argmax per-run accuracy
- positional diagnostics: chosen-letter distribution, mean letter-level probability, accuracy by chosen letter, accuracy by gold-label position, distinct-permutation count
- accuracy by variant type
- accuracy by template and category
- majority-vote accuracy per puzzle
- response stability across seeds (under counterbalancing, this also captures robustness to option-letter permutation)
- cross-variant consistency on majority predictions
- confidence and entropy diagnostics
- cluster bootstrap confidence intervals
- paired variant deltas

## Methodological note

This project is a small controlled diagnostic dataset, not a broad benchmark of logical reasoning.

The goal is to test whether a model preserves the same formal judgment across different narrative variants of the same logical template, while controlling for option-letter positional bias.

For each (puzzle, seed) trial, the pipeline records:

- `option_order`: the per-trial permutation of `(True, False, Unknown, Paradox)` assigned to A/B/C/D
- `sampled_letter` / `sampled_label`: the option letter sampled from the model distribution and the label it maps to under `option_order`
- `argmax_letter` / `argmax_label`: the most probable letter according to the same distribution and the label it maps to
- label-level probabilities (`prob_True`, `prob_False`, `prob_Unknown`, `prob_Paradox`) for logical analysis
- letter-level probabilities (`prob_A`, `prob_B`, `prob_C`, `prob_D`) for positional-bias analysis

`max_prob` and `entropy_nats` are computed over the four-label distribution and are used only as forced-choice option-level diagnostics, not as model-level uncertainty measures.

Decoding defaults are `temperature=0.7` and `top_p=1.0`. Top-p is set to 1.0 (no truncation) because the four-label answer space is not a long-tailed vocabulary: a lower-probability option such as `Unknown` or `Paradox` typically carries meaningful signal rather than vocabulary noise, and `sampled_label` should be drawn from the same posterior that the reported `prob_*` columns describe.
