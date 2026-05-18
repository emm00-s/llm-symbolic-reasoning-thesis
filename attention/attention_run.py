
#!/usr/bin/env python3
"""
attention_run.py

Lightweight attention-map analysis for selected reasoning puzzles.

This script:
- reconstructs the exact behavioral prompt using option_order from the CSV;
- applies the model chat template;
- appends the model's argmax answer letter;
- extracts attention from the answer-token position to prompt tokens;
- averages over heads and the final N layers;
- resolves manually annotated spans;
- aggregates attention by region, role, tag, criticality, and span.

This is diagnostic attention analysis, not causal attribution.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LABELS = ("True", "False", "Unknown", "Paradox")
LETTERS = ("A", "B", "C", "D")

MODEL_NAMES = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def parse_option_order(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        parsed = value
    elif isinstance(value, list):
        parsed = tuple(value)
    else:
        s = str(value).strip()

        if "|" in s:
            parsed = tuple(x.strip() for x in s.split("|"))
        else:
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, (list, tuple)):
                    parsed = tuple(str(x).strip() for x in obj)
                else:
                    raise ValueError
            except Exception:
                parsed = tuple(x.strip().strip("'\"") for x in s.split(","))

    if sorted(parsed) != sorted(LABELS):
        raise ValueError(f"Invalid option_order: {value!r} -> {parsed!r}")

    return parsed


def letter_for_label(label: str, option_order: tuple[str, ...]) -> str:
    return LETTERS[option_order.index(label)]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_results_row(
    rows: list[dict[str, str]],
    puzzle_id: str,
    seed: int,
) -> dict[str, str] | None:
    matches = [r for r in rows if r.get("puzzle_id") == puzzle_id]

    if matches and "seed" in matches[0]:
        matches = [r for r in matches if str(r.get("seed")) == str(seed)]

    if not matches:
        return None

    if len(matches) > 1:
        print(f"[WARN] {puzzle_id} seed={seed}: found {len(matches)} rows; using first.")

    return matches[0]


def load_puzzles_by_id() -> dict[str, dict[str, Any]]:
    from src.dataset import load_puzzles

    return {p["id"]: p for p in load_puzzles()}


def build_raw_prompt(puzzle: dict[str, Any], option_order: tuple[str, ...]) -> str:
    from src.prompt import build_prompt

    return build_prompt(puzzle, option_order=option_order)


def apply_chat_template(raw_prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "spans" in data:
        return {data["puzzle_id"]: data}

    return data


# ---------------------------------------------------------------------
# Span resolver
# ---------------------------------------------------------------------

def find_char_span(prompt: str, span: dict[str, Any]) -> tuple[int, int]:
    text = span["text"]
    anchor = span.get("anchor")
    occurrence = int(span.get("occurrence", 1))

    if anchor:
        anchor_start = prompt.find(anchor)
        if anchor_start == -1:
            raise ValueError(f"Anchor not found: {anchor!r}")

        rel_start = anchor.find(text)
        if rel_start == -1:
            raise ValueError(f"Text {text!r} not found inside anchor {anchor!r}")

        start = anchor_start + rel_start
        end = start + len(text)
        return start, end

    start = -1
    search_from = 0

    for _ in range(occurrence):
        start = prompt.find(text, search_from)
        if start == -1:
            raise ValueError(f"Text not found: {text!r}")
        search_from = start + len(text)

    end = start + len(text)
    return start, end


def token_indices_for_char_span(
    offsets: list[tuple[int, int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    indices = []

    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue

        overlaps = s < end_char and e > start_char
        if overlaps:
            indices.append(i)

    return indices


def resolve_spans(
    *,
    prompt: str,
    tokenizer,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    resolved_spans = []
    errors = []

    by_region: dict[str, list[int]] = {}
    by_role: dict[str, list[int]] = {}
    by_tag: dict[str, list[int]] = {}
    by_critical: dict[str, list[int]] = {}
    by_span: dict[str, list[int]] = {}

    for span in annotation["spans"]:
        try:
            start, end = find_char_span(prompt, span)
            indices = token_indices_for_char_span(offsets, start, end)

            if not indices:
                raise ValueError("No token indices found for span.")

            item = {
                **span,
                "start_char": start,
                "end_char": end,
                "token_indices": indices,
                "tokens": [tokens[i] for i in indices],
            }

            resolved_spans.append(item)
            by_span[span["span_id"]] = indices

            region = span.get("region", "UNSPECIFIED")
            role = span.get("role", "UNSPECIFIED")
            critical = str(bool(span.get("critical", False)))

            by_region.setdefault(region, []).extend(indices)
            by_role.setdefault(role, []).extend(indices)
            by_critical.setdefault(critical, []).extend(indices)

            for tag in span.get("tags", []):
                by_tag.setdefault(tag, []).extend(indices)

        except Exception as e:
            errors.append(
                {
                    "span_id": span.get("span_id"),
                    "text": span.get("text"),
                    "error": str(e),
                }
            )

    # Deduplicate token indices inside each group.
    def dedupe(d: dict[str, list[int]]) -> dict[str, list[int]]:
        return {k: sorted(set(v)) for k, v in d.items()}

    return {
        "resolved_spans": resolved_spans,
        "errors": errors,
        "by_region": dedupe(by_region),
        "by_role": dedupe(by_role),
        "by_tag": dedupe(by_tag),
        "by_critical": dedupe(by_critical),
        "by_span": dedupe(by_span),
    }


# ---------------------------------------------------------------------
# Model + attention extraction
# ---------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {}

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )

    model.eval()
    return model, tokenizer


def extract_answer_attention(
    *,
    model,
    tokenizer,
    chat_prompt: str,
    target_letter: str,
    last_n_layers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_enc = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    full_text = chat_prompt + target_letter

    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = full_enc["input_ids"].to(model.device)
    attention_mask = full_enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    n_prompt_tokens = int(prompt_enc["input_ids"].shape[1])
    n_full_tokens = int(full_enc["input_ids"].shape[1])

    if n_full_tokens <= n_prompt_tokens:
        raise ValueError(
            f"Target letter did not add tokens: prompt={n_prompt_tokens}, full={n_full_tokens}"
        )

    answer_token_index = n_prompt_tokens

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError("No attentions returned. Try attn_implementation='eager'.")

    n_layers = len(attentions)
    start_layer = max(0, n_layers - last_n_layers)
    selected_layers = attentions[start_layer:n_layers]

    layer_vectors = []

    for attn in selected_layers:
        # shape: [batch, heads, seq, seq]
        vec = attn[0, :, answer_token_index, :n_prompt_tokens]
        vec = vec.mean(dim=0)
        layer_vectors.append(vec)

    attention_vec = torch.stack(layer_vectors, dim=0).mean(dim=0)
    attention_vec = attention_vec.detach().float().cpu()

    prompt_ids = prompt_enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids)

    token_attention = []

    for i, (tok, score) in enumerate(zip(tokens, attention_vec.tolist())):
        score = float(score)
        if math.isnan(score) or math.isinf(score):
            continue

        token_attention.append(
            {
                "token_index": i,
                "token": tok,
                "attention": score,
            }
        )

    diagnostics = {
        "n_layers": n_layers,
        "last_n_layers": last_n_layers,
        "used_layers": list(range(start_layer, n_layers)),
        "n_prompt_tokens": n_prompt_tokens,
        "n_full_tokens": n_full_tokens,
        "answer_token_index": answer_token_index,
        "answer_token": tokenizer.convert_ids_to_tokens(
            [full_enc["input_ids"][0, answer_token_index].item()]
        )[0],
        "target_letter": target_letter,
        "attention_sum_to_prompt": float(attention_vec.sum().item()),
    }

    return token_attention, diagnostics


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_attention(
    token_attention: list[dict[str, Any]],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    attn_by_idx = {
        int(item["token_index"]): float(item["attention"])
        for item in token_attention
    }

    def agg(indices: list[int]) -> dict[str, Any]:
        vals = [attn_by_idx[i] for i in indices if i in attn_by_idx]

        n_tokens = len(indices)
        n_scored = len(vals)
        attention_sum = float(sum(vals)) if vals else 0.0
        attention_mean = float(attention_sum / n_scored) if n_scored else 0.0

        return {
            "n_tokens": n_tokens,
            "n_scored_tokens": n_scored,
            "attention_sum": attention_sum,
            "attention_mean_per_token": attention_mean,
        }

    out = {
        "by_region": {},
        "by_role": {},
        "by_tag": {},
        "by_critical": {},
        "by_span": {},
    }

    for key, indices in resolved.get("by_region", {}).items():
        out["by_region"][key] = agg(indices)

    for key, indices in resolved.get("by_role", {}).items():
        out["by_role"][key] = agg(indices)

    for key, indices in resolved.get("by_tag", {}).items():
        out["by_tag"][key] = agg(indices)

    for key, indices in resolved.get("by_critical", {}).items():
        out["by_critical"][key] = agg(indices)

    span_meta = {sp["span_id"]: sp for sp in resolved.get("resolved_spans", [])}

    for span_id, indices in resolved.get("by_span", {}).items():
        meta = span_meta.get(span_id, {})
        out["by_span"][span_id] = {
            **agg(indices),
            "text": meta.get("text"),
            "region": meta.get("region"),
            "role": meta.get("role"),
            "tags": meta.get("tags"),
            "critical": meta.get("critical"),
            "notes": meta.get("notes"),
        }

    return out


def flatten_aggregation(
    *,
    metadata: dict[str, Any],
    aggregation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []

    for group_type in ("by_region", "by_role", "by_tag", "by_critical", "by_span"):
        for group_name, vals in aggregation.get(group_type, {}).items():
            row = {
                **metadata,
                "group_type": group_type.replace("by_", ""),
                "group_name": group_name,
                "n_tokens": vals.get("n_tokens", 0),
                "n_scored_tokens": vals.get("n_scored_tokens", 0),
                "attention_sum": vals.get("attention_sum", 0.0),
                "attention_mean_per_token": vals.get("attention_mean_per_token", 0.0),
            }

            if group_type == "by_span":
                row["span_text"] = vals.get("text")
                row["span_region"] = vals.get("region")
                row["span_role"] = vals.get("role")
                row["span_tags"] = "|".join(vals.get("tags") or [])
                row["span_critical"] = vals.get("critical")
                row["span_notes"] = vals.get("notes")

            rows.append(row)

    return rows


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        print(f"[WARN] No rows to save: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


# ---------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------

def run_one(
    *,
    model_key: str,
    model,
    tokenizer,
    puzzle: dict[str, Any],
    row: dict[str, str],
    annotation: dict[str, Any],
    seed: int,
    last_n_layers: int,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    puzzle_id = puzzle["id"]
    option_order = parse_option_order(row["option_order"])

    gold_label = row["gold_label"]
    gold_letter = letter_for_label(gold_label, option_order)

    argmax_label = row["argmax_label"]
    argmax_letter = row["argmax_letter"]

    raw_prompt = build_raw_prompt(puzzle, option_order)
    chat_prompt = apply_chat_template(raw_prompt, tokenizer)

    resolved = resolve_spans(
        prompt=chat_prompt,
        tokenizer=tokenizer,
        annotation=annotation,
    )

    if resolved["errors"]:
        print(f"[ERROR] Span resolution failed for {puzzle_id}")
        for err in resolved["errors"]:
            print(" ", err)
        return None, []

    token_attention, diagnostics = extract_answer_attention(
        model=model,
        tokenizer=tokenizer,
        chat_prompt=chat_prompt,
        target_letter=argmax_letter,
        last_n_layers=last_n_layers,
    )

    aggregation = aggregate_attention(token_attention, resolved)

    metadata = {
        "model_key": model_key,
        "model_name": MODEL_NAMES[model_key],
        "puzzle_id": puzzle_id,
        "template_id": puzzle["template_id"],
        "variant_type": puzzle["variant_type"],
        "seed": seed,
        "option_order": "|".join(option_order),
        "gold_label": gold_label,
        "gold_letter": gold_letter,
        "argmax_label": argmax_label,
        "argmax_letter": argmax_letter,
        "target_letter": argmax_letter,
        "correct_argmax": str(argmax_label == gold_label),
        "last_n_layers": last_n_layers,
    }

    result = {
        "metadata": metadata,
        "diagnostics": diagnostics,
        "raw_prompt": raw_prompt,
        "chat_prompt": chat_prompt,
        "resolved": resolved,
        "token_attention": token_attention,
        "aggregation": aggregation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{puzzle_id}_{model_key}_seed{seed}_attention"
    json_path = json_dir / f"{stem}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"{model_key:5s} | {puzzle_id:24s} | "
        f"gold={gold_label}->{gold_letter} | "
        f"argmax={argmax_label}->{argmax_letter} | "
        f"answer_token={diagnostics['answer_token']!r} | "
        f"attn_sum={diagnostics['attention_sum_to_prompt']:.4f}"
    )

    print("  TAGS:")
    for tag, vals in aggregation["by_tag"].items():
        print(
            f"    {tag:16s} "
            f"sum={vals['attention_sum']:.6f} "
            f"mean={vals['attention_mean_per_token']:.6f} "
            f"n={vals['n_scored_tokens']}"
        )

    flat_rows = flatten_aggregation(metadata=metadata, aggregation=aggregation)
    return result, flat_rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--template-id", required=True)
    parser.add_argument("--models", nargs="+", choices=["qwen", "llama"], default=["qwen"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--qwen-csv", type=Path)
    parser.add_argument("--llama-csv", type=Path)
    parser.add_argument("--spans-json", type=Path, required=True)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("attention_outputs/batch"))

    args = parser.parse_args()

    annotations = load_annotations(args.spans_json)
    puzzles_by_id = load_puzzles_by_id()

    selected_puzzle_ids = [
        pid for pid, ann in annotations.items()
        if ann.get("template_id") == args.template_id
    ]

    selected_puzzle_ids = sorted(selected_puzzle_ids)

    if not selected_puzzle_ids:
        raise ValueError(f"No annotated puzzles found for template {args.template_id}")

    print(f"Selected puzzles: {selected_puzzle_ids}")

    csv_paths = {
        "qwen": args.qwen_csv,
        "llama": args.llama_csv,
    }

    csv_rows = {}

    for model_key in args.models:
        csv_path = csv_paths[model_key]
        if csv_path is None:
            raise ValueError(f"--{model_key}-csv is required for model {model_key}")
        csv_rows[model_key] = load_csv_rows(csv_path)

    all_rows = []
    summary = []

    for model_key in args.models:
        model_name = MODEL_NAMES[model_key]

        print("\n" + "#" * 88)
        print(f"Loading model: {model_key} | {model_name}")
        print("#" * 88)

        model, tokenizer = load_model_and_tokenizer(model_name)

        for seed in args.seeds:
            for puzzle_id in selected_puzzle_ids:
                puzzle = puzzles_by_id[puzzle_id]
                annotation = annotations[puzzle_id]
                row = load_results_row(csv_rows[model_key], puzzle_id, seed)

                if row is None:
                    print(f"[WARN] Missing CSV row: {model_key}, {puzzle_id}, seed={seed}")
                    continue

                result, flat_rows = run_one(
                    model_key=model_key,
                    model=model,
                    tokenizer=tokenizer,
                    puzzle=puzzle,
                    row=row,
                    annotation=annotation,
                    seed=seed,
                    last_n_layers=args.last_n_layers,
                    output_dir=args.output_dir,
                )

                all_rows.extend(flat_rows)

                if result is not None:
                    summary.append(result["metadata"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_csv = args.output_dir / f"{args.template_id}_attention_aggregated.csv"
    save_csv(all_rows, out_csv)

    summary_path = args.output_dir / f"{args.template_id}_attention_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
