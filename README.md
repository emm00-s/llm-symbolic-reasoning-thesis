cat > README.md <<'EOF'
# LLM Symbolic Reasoning Thesis

Controlled diagnostic benchmark for evaluating logical invariance in language models under narrative variation.

## Project structure

- `puzzles/puzzles.json`: natural-language puzzle variants
- `puzzles/templates.json`: formal templates, gold labels, and Z3 verification code
- `src/dataset.py`: loads and joins puzzles with templates
- `src/solvers.py`: verifies gold labels with Z3
- `src/prompt.py`: builds A/B/C/D prompts and parses answers
- `src/model.py`: wraps Qwen-2.5-3B-Instruct and samples answers from the model distribution over A/B/C/D
- `src/run.py`: runs repeated sampled evaluations
- `src/analyze.py`: computes accuracy, majority vote, stability, cross-variant consistency, and confidence diagnostics

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

## Label mapping

The task uses a four-option multiple-choice format:

| Option | Label |
|---|---|
| A | True |
| B | False |
| C | Unknown |
| D | Paradox |

## Execution mode

The `src/` directory is a Python package.

All scripts should be run from the repository root using module mode:

```bash
python -m src.dataset
python -m src.solvers
python -m src.run --seeds 10 --temperature 0.7 --top-p 0.9 --tag qwen3b
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
python -m src.run --seeds 2 --temperature 0.7 --top-p 0.9 --tag smoke
```

Full run:

```bash
python -m src.run --seeds 10 --temperature 0.7 --top-p 0.9 --tag qwen3b
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
- majority-vote accuracy
- response stability across seeds
- accuracy by variant type
- accuracy by template and category
- cross-variant consistency on majority predictions
- confidence and entropy diagnostics
- cluster bootstrap confidence intervals
- paired variant deltas

## Methodological note

This project is a small controlled diagnostic dataset, not a broad benchmark of logical reasoning.

The goal is to test whether a model preserves the same formal judgment across different narrative variants of the same logical template.

For each puzzle, the pipeline records both:

- `sampled_label`: the label sampled from the model distribution over A/B/C/D
- `argmax_label`: the most probable label according to the same distribution

Confidence and entropy are computed over the normalized option probabilities and are used only as option-level diagnostics.
EOF