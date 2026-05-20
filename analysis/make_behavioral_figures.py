"""Generate the behavioral analysis figures for Qwen-2.5-3B vs Llama-3.2-3B.

Reads the two result CSVs from ``results/`` and writes PNGs (and one CSV
table) to ``figures/behavioral/``.

Dependencies (not installed automatically by this script):
    pip install pandas plotly kaleido==0.2.1

Usage:
    python analysis/make_behavioral_figures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures" / "behavioral"

QWEN_CSV = RESULTS_DIR / "results_qwen3b_T0.7_topP1.0_n10_20260513_114931.csv"
LLAMA_CSV = RESULTS_DIR / "results_llama32_3b_T0.7_topP1.0_n10_20260513_154349.csv"

LABEL_ORDER = ["True", "False", "Unknown", "Paradox"]
VARIANT_ORDER = ["abstract", "artificial", "familiar", "belief_violating"]
MODEL_ORDER = ["Qwen-2.5-3B", "Llama-3.2-3B"]
LETTER_COLS = ["prob_A", "prob_B", "prob_C", "prob_D"]
LETTER_ORDER = ["A", "B", "C", "D"]


def short_model_name(model_name: str) -> str:
    if "Qwen" in model_name:
        return "Qwen-2.5-3B"
    if "Llama" in model_name:
        return "Llama-3.2-3B"
    return model_name


def to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() == "true"


def load_data() -> pd.DataFrame:
    qwen = pd.read_csv(QWEN_CSV)
    llama = pd.read_csv(LLAMA_CSV)

    if "model_name" not in qwen.columns:
        qwen["model_name"] = "Qwen/Qwen2.5-3B-Instruct"
    if "model_name" not in llama.columns:
        llama["model_name"] = "meta-llama/Llama-3.2-3B-Instruct"

    df = pd.concat([qwen, llama], ignore_index=True)
    df["model_short"] = df["model_name"].apply(short_model_name)
    df["correct_sampled"] = df["correct_sampled"].apply(to_bool)
    df["correct_argmax"] = df["correct_argmax"].apply(to_bool)
    df["argmax_correctness"] = df["correct_argmax"].map({True: "Correct", False: "Incorrect"})
    return df


def save(fig, name: str, *, html: bool = False) -> None:
    fig.write_image(str(FIG_DIR / f"{name}.png"), scale=3)
    if html:
        fig.write_html(str(FIG_DIR / f"{name}.html"))


def fig_overall_accuracy(df: pd.DataFrame) -> None:
    overall = (
        df.groupby("model_short")
          .agg(sampled_accuracy=("correct_sampled", "mean"),
               argmax_accuracy=("correct_argmax", "mean"),
               n=("correct_argmax", "size"))
          .reset_index()
    )
    overall["sampled_accuracy"] *= 100
    overall["argmax_accuracy"] *= 100

    overall_long = overall.melt(
        id_vars=["model_short", "n"],
        value_vars=["sampled_accuracy", "argmax_accuracy"],
        var_name="metric", value_name="accuracy",
    )
    overall_long["metric"] = overall_long["metric"].map({
        "sampled_accuracy": "Sampled accuracy",
        "argmax_accuracy": "Argmax accuracy",
    })

    fig = px.bar(
        overall_long, x="model_short", y="accuracy", color="metric", barmode="group",
        text=overall_long["accuracy"].round(1).astype(str) + "%",
        labels={"model_short": "Model", "accuracy": "Accuracy (%)", "metric": "Metric"},
        title="Overall forced-choice accuracy by model", template="plotly_white",
    )
    fig.update_traces(textposition="outside", marker_line_width=0.8)
    fig.update_layout(
        width=850, height=520,
        title={"text": "Overall forced-choice accuracy by model",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14}, legend_title_text="",
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), bargap=0.25, bargroupgap=0.08,
    )
    save(fig, "fig_01_overall_accuracy_by_model")


def fig_accuracy_by_variant(df: pd.DataFrame) -> None:
    variant_acc = (
        df.groupby(["model_short", "variant_type"])
          .agg(argmax_accuracy=("correct_argmax", "mean"),
               sampled_accuracy=("correct_sampled", "mean"),
               n=("correct_argmax", "size"))
          .reset_index()
    )
    variant_acc["argmax_accuracy"] *= 100
    variant_acc["sampled_accuracy"] *= 100
    variant_acc["variant_type"] = pd.Categorical(
        variant_acc["variant_type"], categories=VARIANT_ORDER, ordered=True
    )
    variant_acc = variant_acc.sort_values(["variant_type", "model_short"])

    for metric, name, ylabel in [
        ("argmax_accuracy", "fig_02_accuracy_by_narrative_variant", "Argmax accuracy (%)"),
        ("sampled_accuracy", "fig_02_sampled_accuracy_by_narrative_variant", "Sampled accuracy (%)"),
    ]:
        fig = px.bar(
            variant_acc, x="variant_type", y=metric, color="model_short", barmode="group",
            text=variant_acc[metric].round(1).astype(str) + "%",
            category_orders={"variant_type": VARIANT_ORDER, "model_short": MODEL_ORDER},
            labels={"variant_type": "Narrative variant", metric: ylabel, "model_short": "Model"},
            title=name.replace("_", " "), template="plotly_white",
        )
        fig.update_traces(textposition="outside", marker_line_width=0.8)
        fig.update_layout(
            width=950, height=560,
            title={"text": ("Argmax" if metric == "argmax_accuracy" else "Sampled")
                   + " accuracy by narrative variant",
                   "x": 0.5, "xanchor": "center", "font": {"size": 22}},
            font={"size": 14}, legend_title_text="",
            yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
            xaxis=dict(title="", tickangle=0),
            bargap=0.25, bargroupgap=0.08,
        )
        save(fig, name)


def fig_template_variant_heatmap(df: pd.DataFrame) -> None:
    heatmap_data = (
        df.groupby(["model_short", "template_id", "variant_type"])
          .agg(argmax_accuracy=("correct_argmax", "mean"),
               n=("correct_argmax", "size"))
          .reset_index()
    )
    heatmap_data["argmax_accuracy"] *= 100
    template_order = sorted(df["template_id"].unique())

    for model, fname in [("Qwen-2.5-3B", "fig_03a_qwen_template_variant_heatmap"),
                         ("Llama-3.2-3B", "fig_03b_llama_template_variant_heatmap")]:
        model_data = heatmap_data[heatmap_data["model_short"] == model]
        pivot = model_data.pivot(index="template_id", columns="variant_type",
                                 values="argmax_accuracy")
        pivot = pivot.reindex(index=template_order, columns=VARIANT_ORDER)

        fig = px.imshow(
            pivot, text_auto=".1f", color_continuous_scale="RdYlGn",
            zmin=0, zmax=100,
            labels=dict(x="Narrative variant", y="Template", color="Accuracy (%)"),
            title=f"{model}: argmax accuracy by template and narrative variant",
            template="plotly_white", aspect="auto",
        )
        fig.update_traces(texttemplate="%{z:.1f}%", textfont={"size": 13})
        fig.update_layout(
            width=900, height=620,
            title={"x": 0.5, "xanchor": "center", "font": {"size": 22}},
            font={"size": 14}, xaxis=dict(side="bottom"),
            coloraxis_colorbar=dict(title="Accuracy (%)", ticksuffix="%"),
        )
        save(fig, fname)


def fig_accuracy_by_template(df: pd.DataFrame) -> None:
    template_acc = (
        df.groupby(["model_short", "template_id"])
          .agg(argmax_accuracy=("correct_argmax", "mean"),
               n=("correct_argmax", "size"))
          .reset_index()
    )
    template_acc["argmax_accuracy"] *= 100
    template_acc = template_acc.sort_values(["template_id", "model_short"])
    template_order = sorted(df["template_id"].unique())

    fig = px.bar(
        template_acc, x="template_id", y="argmax_accuracy", color="model_short",
        barmode="group",
        text=template_acc["argmax_accuracy"].round(1).astype(str) + "%",
        category_orders={"template_id": template_order, "model_short": MODEL_ORDER},
        labels={"template_id": "Logical template",
                "argmax_accuracy": "Argmax accuracy (%)", "model_short": "Model"},
        title="Argmax accuracy by logical template", template="plotly_white",
    )
    fig.update_traces(textposition="outside", marker_line_width=0.8)
    fig.update_layout(
        width=1050, height=560,
        title={"text": "Argmax accuracy by logical template",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14}, legend_title_text="",
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title="Logical template"),
        bargap=0.25, bargroupgap=0.08,
    )
    save(fig, "fig_04_accuracy_by_template")

    fig_slope = px.line(
        template_acc, x="model_short", y="argmax_accuracy", color="template_id",
        markers=True,
        labels={"model_short": "Model", "argmax_accuracy": "Argmax accuracy (%)",
                "template_id": "Template"},
        title="Template-level accuracy shift from Qwen to Llama",
        template="plotly_white",
        category_orders={"model_short": MODEL_ORDER},
    )
    fig_slope.update_traces(line=dict(width=2), marker=dict(size=9))
    fig_slope.update_layout(
        width=850, height=600,
        title={"x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14},
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), legend_title_text="Template",
    )
    save(fig_slope, "fig_04b_template_accuracy_slopeplot")


def fig_label_distribution(df: pd.DataFrame) -> None:
    pred_dist = (
        df.groupby(["model_short", "argmax_label"]).size().reset_index(name="count")
    )
    pred_dist["percentage"] = (
        pred_dist.groupby("model_short")["count"].transform(lambda x: x / x.sum() * 100)
    )
    pred_dist["argmax_label"] = pd.Categorical(
        pred_dist["argmax_label"], categories=LABEL_ORDER, ordered=True
    )
    pred_dist = pred_dist.sort_values(["argmax_label", "model_short"])

    fig = px.bar(
        pred_dist, x="argmax_label", y="percentage", color="model_short", barmode="group",
        text=pred_dist["percentage"].round(1).astype(str) + "%",
        category_orders={"argmax_label": LABEL_ORDER, "model_short": MODEL_ORDER},
        labels={"argmax_label": "Predicted label", "percentage": "Predictions (%)",
                "model_short": "Model"},
        title="Distribution of argmax predictions by model", template="plotly_white",
    )
    fig.update_traces(textposition="outside", marker_line_width=0.8)
    fig.update_layout(
        width=900, height=560,
        title={"text": "Distribution of argmax predictions by model",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14}, legend_title_text="",
        yaxis=dict(range=[0, 60], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), bargap=0.25, bargroupgap=0.08,
    )
    save(fig, "fig_05_argmax_label_distribution")

    gold_dist = (
        df.drop_duplicates("puzzle_id").groupby("gold_label").size().reset_index(name="count")
    )
    gold_dist["percentage"] = gold_dist["count"] / gold_dist["count"].sum() * 100
    gold_dist["source"] = "Gold labels"
    gold_dist = gold_dist.rename(columns={"gold_label": "label"})

    pred = pred_dist.rename(columns={"argmax_label": "label"}).copy()
    pred["source"] = pred["model_short"]

    label_comparison = pd.concat(
        [gold_dist[["label", "percentage", "source"]],
         pred[["label", "percentage", "source"]]],
        ignore_index=True,
    )
    label_comparison["label"] = pd.Categorical(
        label_comparison["label"], categories=LABEL_ORDER, ordered=True
    )

    fig = px.bar(
        label_comparison, x="label", y="percentage", color="source", barmode="group",
        text=label_comparison["percentage"].round(1).astype(str) + "%",
        category_orders={"label": LABEL_ORDER,
                         "source": ["Gold labels", *MODEL_ORDER]},
        labels={"label": "Label", "percentage": "Percentage (%)", "source": ""},
        title="Gold-label distribution vs model predictions", template="plotly_white",
    )
    fig.update_traces(textposition="outside", marker_line_width=0.8)
    fig.update_layout(
        width=950, height=560,
        title={"text": "Gold-label distribution vs model predictions",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14},
        yaxis=dict(range=[0, 60], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), bargap=0.25, bargroupgap=0.08,
    )
    save(fig, "fig_05_gold_vs_predicted_labels")


def fig_confusion_matrices(df: pd.DataFrame) -> None:
    for model, fname in [("Qwen-2.5-3B", "fig_06a_qwen_confusion_matrix"),
                         ("Llama-3.2-3B", "fig_06b_llama_confusion_matrix")]:
        model_df = df[df["model_short"] == model]
        cm = pd.crosstab(model_df["gold_label"], model_df["argmax_label"],
                         normalize="index") * 100
        cm = cm.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

        fig = px.imshow(
            cm, text_auto=".1f", color_continuous_scale="Blues",
            zmin=0, zmax=100,
            labels=dict(x="Predicted label", y="Gold label", color="Percentage (%)"),
            title=f"{model}: row-normalized confusion matrix",
            template="plotly_white", aspect="auto",
        )
        fig.update_traces(texttemplate="%{z:.1f}%", textfont={"size": 14})
        fig.update_layout(
            width=720, height=620,
            title={"text": f"{model}: row-normalized confusion matrix",
                   "x": 0.5, "xanchor": "center", "font": {"size": 21}},
            font={"size": 14}, xaxis=dict(side="bottom"),
            coloraxis_colorbar=dict(title="%", ticksuffix="%"),
        )
        save(fig, fname)


def fig_max_probability(df: pd.DataFrame) -> None:
    common = dict(
        x="model_short", y="max_prob", color="argmax_correctness", points="all",
        category_orders={"model_short": MODEL_ORDER,
                         "argmax_correctness": ["Correct", "Incorrect"]},
        labels={"model_short": "Model", "max_prob": "Forced-choice max probability",
                "argmax_correctness": ""},
        template="plotly_white",
    )

    fig = px.box(df, title="Forced-choice max probability by correctness", **common)
    fig.update_layout(
        width=900, height=600,
        title={"text": "Forced-choice max probability by correctness",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14},
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), legend_title_text="",
    )
    save(fig, "fig_07_max_probability_by_correctness")

    fig_v = px.violin(df, box=True, title="Distribution concentration by correctness", **common)
    fig_v.update_layout(
        width=900, height=600,
        title={"text": "Distribution concentration by correctness",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14},
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), legend_title_text="",
    )
    save(fig_v, "fig_07b_max_probability_violin")


def fig_entropy(df: pd.DataFrame) -> None:
    common = dict(
        x="model_short", y="entropy_nats", color="argmax_correctness", points="all",
        category_orders={"model_short": MODEL_ORDER,
                         "argmax_correctness": ["Correct", "Incorrect"]},
        labels={"model_short": "Model", "entropy_nats": "Entropy (nats)",
                "argmax_correctness": ""},
        template="plotly_white",
    )

    fig = px.box(df, title="Forced-choice entropy by correctness", **common)
    fig.update_layout(
        width=900, height=600,
        title={"text": "Forced-choice entropy by correctness",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14},
        yaxis=dict(range=[0, 1.45], gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), legend_title_text="",
    )
    save(fig, "fig_08_entropy_by_correctness")

    fig_v = px.violin(df, box=True, title="Distributional entropy by correctness", **common)
    fig_v.update_layout(
        width=900, height=600,
        title={"text": "Distributional entropy by correctness",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14},
        yaxis=dict(range=[0, 1.45], gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), legend_title_text="",
    )
    save(fig_v, "fig_08b_entropy_violin")


def fig_letter_bias(df: pd.DataFrame) -> None:
    letter_prob = df.groupby("model_short")[LETTER_COLS].mean().reset_index()
    letter_prob_long = letter_prob.melt(
        id_vars="model_short", value_vars=LETTER_COLS,
        var_name="letter", value_name="mean_probability",
    )
    letter_prob_long["letter"] = letter_prob_long["letter"].str.replace("prob_", "")
    letter_prob_long["mean_probability_pct"] = letter_prob_long["mean_probability"] * 100

    fig = px.bar(
        letter_prob_long, x="letter", y="mean_probability_pct",
        color="model_short", barmode="group",
        text=letter_prob_long["mean_probability_pct"].round(1).astype(str) + "%",
        category_orders={"letter": LETTER_ORDER, "model_short": MODEL_ORDER},
        labels={"letter": "Answer letter",
                "mean_probability_pct": "Mean probability (%)", "model_short": "Model"},
        title="Mean forced-choice probability by answer letter",
        template="plotly_white",
    )
    fig.add_hline(y=25, line_dash="dash", line_color="gray",
                  annotation_text="Uniform baseline: 25%", annotation_position="top right")
    fig.update_traces(textposition="outside", marker_line_width=0.8)
    fig.update_layout(
        width=900, height=560,
        title={"text": "Mean forced-choice probability by answer letter",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14}, legend_title_text="",
        yaxis=dict(range=[0, 50], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), bargap=0.25, bargroupgap=0.08,
    )
    save(fig, "fig_09_mean_probability_by_answer_letter")

    argmax_letter_dist = (
        df.groupby(["model_short", "argmax_letter"]).size().reset_index(name="count")
    )
    argmax_letter_dist["percentage"] = (
        argmax_letter_dist.groupby("model_short")["count"]
                          .transform(lambda x: x / x.sum() * 100)
    )
    argmax_letter_dist["argmax_letter"] = pd.Categorical(
        argmax_letter_dist["argmax_letter"], categories=LETTER_ORDER, ordered=True
    )

    fig = px.bar(
        argmax_letter_dist, x="argmax_letter", y="percentage",
        color="model_short", barmode="group",
        text=argmax_letter_dist["percentage"].round(1).astype(str) + "%",
        category_orders={"argmax_letter": LETTER_ORDER, "model_short": MODEL_ORDER},
        labels={"argmax_letter": "Argmax letter",
                "percentage": "Argmax selections (%)", "model_short": "Model"},
        title="Distribution of argmax answer letters", template="plotly_white",
    )
    fig.add_hline(y=25, line_dash="dash", line_color="gray",
                  annotation_text="Uniform baseline: 25%", annotation_position="top right")
    fig.update_traces(textposition="outside", marker_line_width=0.8)
    fig.update_layout(
        width=900, height=560,
        title={"text": "Distribution of argmax answer letters",
               "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        font={"size": 14}, legend_title_text="",
        yaxis=dict(range=[0, 55], ticksuffix="%", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(title=""), bargap=0.25, bargroupgap=0.08,
    )
    save(fig, "fig_09b_argmax_letter_distribution")


def fig_cross_variant_majority(df: pd.DataFrame) -> None:
    template_order = sorted(df["template_id"].unique())

    def majority_label(series):
        return series.value_counts().index[0]

    majority = (
        df.groupby(["model_short", "template_id", "variant_type"])
          .agg(majority_argmax_label=("argmax_label", majority_label),
               gold_label=("gold_label", "first"),
               n=("argmax_label", "size"))
          .reset_index()
    )
    majority["majority_correct"] = majority["majority_argmax_label"] == majority["gold_label"]

    consistency = (
        majority.groupby(["model_short", "template_id"])
          .agg(n_unique_predictions=("majority_argmax_label", "nunique"),
               predictions=("majority_argmax_label", lambda x: " | ".join(x)),
               all_variants_same=("majority_argmax_label", lambda x: x.nunique() == 1),
               gold_label=("gold_label", "first"))
          .reset_index()
    )
    consistency["status"] = consistency["all_variants_same"].map(
        {True: "consistent", False: "variant-sensitive"}
    )
    consistency[["model_short", "template_id", "gold_label", "predictions", "status"]] \
        .to_csv(FIG_DIR / "table_10_cross_variant_consistency.csv", index=False)

    label_to_code = {"True": 0, "False": 1, "Unknown": 2, "Paradox": 3}
    color_scale = [
        [0.00, "#4C78A8"], [0.24, "#4C78A8"],
        [0.25, "#F58518"], [0.49, "#F58518"],
        [0.50, "#54A24B"], [0.74, "#54A24B"],
        [0.75, "#B279A2"], [1.00, "#B279A2"],
    ]

    for model, fname in [("Qwen-2.5-3B", "fig_10a_qwen_cross_variant_majority_prediction"),
                         ("Llama-3.2-3B", "fig_10b_llama_cross_variant_majority_prediction")]:
        model_data = majority[majority["model_short"] == model].copy()
        model_data["label_code"] = model_data["majority_argmax_label"].map(label_to_code)

        pivot_code = model_data.pivot(index="template_id", columns="variant_type",
                                      values="label_code") \
                               .reindex(index=template_order, columns=VARIANT_ORDER)
        pivot_text = model_data.pivot(index="template_id", columns="variant_type",
                                      values="majority_argmax_label") \
                               .reindex(index=template_order, columns=VARIANT_ORDER)

        fig = px.imshow(
            pivot_code, text_auto=False, color_continuous_scale=color_scale,
            zmin=0, zmax=3,
            labels=dict(x="Narrative variant", y="Template", color="Majority label"),
            title=f"{model}: majority argmax prediction across variants",
            template="plotly_white", aspect="auto",
        )
        fig.update_traces(text=pivot_text.values, texttemplate="%{text}",
                          textfont={"size": 13})
        fig.update_layout(
            width=980, height=620,
            title={"text": f"{model}: majority argmax prediction across variants",
                   "x": 0.5, "xanchor": "center", "font": {"size": 22}},
            font={"size": 14}, xaxis=dict(side="bottom"),
            coloraxis_colorbar=dict(title="Label", tickvals=[0, 1, 2, 3],
                                    ticktext=LABEL_ORDER),
        )
        save(fig, fname)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows from {QWEN_CSV.name} + {LLAMA_CSV.name}")

    fig_overall_accuracy(df)
    fig_accuracy_by_variant(df)
    fig_template_variant_heatmap(df)
    fig_accuracy_by_template(df)
    fig_label_distribution(df)
    fig_confusion_matrices(df)
    fig_max_probability(df)
    fig_entropy(df)
    fig_letter_bias(df)
    fig_cross_variant_majority(df)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
