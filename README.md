# LLM Symbolic Reasoning

Controlled diagnostic benchmark for evaluating logical invariance in language models under narrative variation.

## Structure

- `puzzles/puzzles.json`: natural-language puzzle variants
- `puzzles/templates.json`: formal templates, gold labels, and Z3 verification code
- `src/dataset.py`: loads and joins puzzles with templates
- `src/solvers.py`: verifies gold labels with Z3
- `src/prompt.py`: builds A/B/C/D prompts and parses answers
- `src/model.py`: wraps Qwen-2.5-3B-Instruct
- `src/run.py`: runs repeated sampled evaluations
- `src/analyze.py`: computes accuracy, stability, consistency, and confidence diagnostics

## Label mapping

A = True  
B = False  
C = Unknown  
D = Paradox