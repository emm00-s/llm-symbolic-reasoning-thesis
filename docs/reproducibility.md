# Reproducibility

This document describes how to regenerate the final behavioral artifacts
(CSV summaries and figures) that compare Qwen-2.5-3B-Instruct and
Llama-3.2-3B-Instruct on the logical-invariance benchmark.

## Final result CSVs

The two committed CSVs in `results/` are the canonical model outputs:

- `results/results_qwen3b_T0.7_topP1.0_n10_20260513_114931.csv`
- `results/results_llama32_3b_T0.7_topP1.0_n10_20260513_154349.csv`

Each is the output of `src.run` with `--seeds 10 --temperature 0.7 --top-p 1.0`.

## Regenerating the per-model text summaries

```bash
python -m src.analyze results/results_qwen3b_T0.7_topP1.0_n10_20260513_114931.csv > results/analysis_qwen3b.txt
python -m src.analyze results/results_llama32_3b_T0.7_topP1.0_n10_20260513_154349.csv > results/analysis_llama32_3b.txt
```

## Regenerating the figures

```bash
python analysis/make_behavioral_figures.py
```

This writes all PNGs (and `table_10_cross_variant_consistency.csv`) into
`figures/behavioral/`. The script depends on `pandas`, `plotly`, and
`kaleido==0.2.1`, all of which are pinned in `requirements.txt`.

## Re-running the benchmark from scratch

The Colab-style workflow used to produce the two CSVs is documented in
`analysis/run_benchmark_colab.py`. The full Qwen and Llama runs require a GPU.
`meta-llama/Llama-3.2-3B-Instruct` is a gated model, so accept its license on
Hugging Face and authenticate (`huggingface-cli login` or `HF_TOKEN`) before
running.

```bash
pip install -r requirements.txt
python -m src.dataset
python -m src.solvers
python -m src.run --model-name Qwen/Qwen2.5-3B-Instruct --seeds 10 --temperature 0.7 --top-p 1.0 --tag qwen3b
python -m src.run --model-name meta-llama/Llama-3.2-3B-Instruct --seeds 10 --temperature 0.7 --top-p 1.0 --tag llama32_3b
```
