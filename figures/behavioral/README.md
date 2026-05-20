# Behavioral figures

Figures comparing Qwen-2.5-3B-Instruct and Llama-3.2-3B-Instruct on the
forced-choice logical-invariance benchmark. All PNGs are generated from the
two final result CSVs in `results/`.

## Regenerate

From the repository root:

```bash
python analysis/make_behavioral_figures.py
```

Requires `pandas`, `plotly`, and `kaleido==0.2.1` (see `requirements.txt`).

## Contents

| File | Description |
| --- | --- |
| `fig_01_overall_accuracy_by_model.png` | Overall sampled vs. argmax accuracy per model. |
| `fig_02_accuracy_by_narrative_variant.png` | Argmax accuracy split by narrative variant (abstract / artificial / familiar / belief-violating). |
| `fig_02_sampled_accuracy_by_narrative_variant.png` | Same as above but for sampled accuracy. |
| `fig_03a_qwen_template_variant_heatmap.png` | Qwen heatmap: argmax accuracy by template x narrative variant. |
| `fig_03b_llama_template_variant_heatmap.png` | Llama heatmap: argmax accuracy by template x narrative variant. |
| `fig_04_accuracy_by_template.png` | Grouped barplot of argmax accuracy per logical template. |
| `fig_04b_template_accuracy_slopeplot.png` | Slope plot showing per-template accuracy shift from Qwen to Llama. |
| `fig_05_argmax_label_distribution.png` | Distribution of argmax label predictions per model. |
| `fig_05_gold_vs_predicted_labels.png` | Gold-label distribution vs each model's predicted distribution. |
| `fig_06a_qwen_confusion_matrix.png` | Qwen row-normalized confusion matrix (gold x predicted). |
| `fig_06b_llama_confusion_matrix.png` | Llama row-normalized confusion matrix. |
| `fig_07_max_probability_by_correctness.png` | Boxplot of forced-choice max probability, split by correctness. |
| `fig_07b_max_probability_violin.png` | Violin version of the same comparison. |
| `fig_08_entropy_by_correctness.png` | Boxplot of forced-choice entropy (nats), split by correctness. |
| `fig_08b_entropy_violin.png` | Violin version of the same comparison. |
| `fig_09_mean_probability_by_answer_letter.png` | Mean probability assigned to each option letter A/B/C/D. |
| `fig_09b_argmax_letter_distribution.png` | Distribution of argmax letters (positional bias check). |
| `fig_10a_qwen_cross_variant_majority_prediction.png` | Qwen majority argmax prediction across narrative variants per template. |
| `fig_10b_llama_cross_variant_majority_prediction.png` | Llama majority argmax prediction across narrative variants per template. |
| `table_10_cross_variant_consistency.csv` | Per-template cross-variant consistency status (consistent vs variant-sensitive). |
