"""Local model wrapper around Qwen-2.5-3B-Instruct.

call_llm(prompt, seed, temperature, top_p) performs one constrained sampled
answer selection over the admissible labels A/B/C/D.

It returns:
  - raw_text: canonical generated answer letter, e.g. "A"
  - sampled_label: the label actually sampled from the model distribution
  - argmax_label: the most probable label according to the answer-option logprobs
  - first_token_logprobs: dict {label: log_prob} over the 4 answer labels
  - first_token_probs: dict {label: normalized probability} over the 4 labels

The model is sampled multiple times with different seeds to measure response
variability across repeated runs. The logprobs are computed at the answer step
and are aggregated over tokenization variants such as "A", " A", and "\\nA".

Auto-detects CUDA / MPS / CPU. Model is loaded once with lru_cache.
"""

import math
from functools import lru_cache

from .prompt import LABELS, LABEL_LETTER, LETTER_LABEL

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def _logsumexp(values: list[float]) -> float:
    """Numerically stable log(sum(exp(v))) over log-values."""
    if not values:
        return float("-inf")

    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _renormalize(logprobs: dict[str, float]) -> dict[str, float]:
    """Renormalize logprobs over the four answer labels."""
    missing = [label for label in LABELS if label not in logprobs]

    if missing:
        raise ValueError(f"Missing logprobs for labels: {missing}")

    m = max(logprobs[label] for label in LABELS)

    if m == float("-inf"):
        raise ValueError("All label logprobs are -inf. Check answer token IDs.")

    exps = {label: math.exp(logprobs[label] - m) for label in LABELS}
    total = sum(exps.values())

    return {label: exps[label] / total for label in LABELS}


def _apply_top_p(probs: dict[str, float], top_p: float) -> dict[str, float]:
    """Apply nucleus filtering over the four label probabilities.

    Filtering is done at label level, not token level, because each label may
    correspond to multiple tokenization variants.
    """
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_items = sorted(probs.items(), key=lambda item: item[1], reverse=True)

    kept: list[tuple[str, float]] = []
    cumulative = 0.0

    for label, prob in sorted_items:
        kept.append((label, prob))
        cumulative += prob

        if cumulative >= top_p:
            break

    total = sum(prob for _, prob in kept)

    return {label: prob / total for label, prob in kept}


@lru_cache(maxsize=1)
def _load():
    """Load model and tokenizer once.

    Returns:
        tokenizer, model, device, letter_token_ids
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"loading {MODEL_NAME} on {device} ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    # Collect single-token IDs for each answer letter.
    # Leading-space and newline-prefixed variants are included because chat
    # models may naturally start answers with whitespace or a newline.
    letter_token_ids: dict[str, list[int]] = {}

    for letter, label in LETTER_LABEL.items():
        token_ids = set()

        for variant in (letter, f" {letter}", f"\n{letter}", letter.lower(), f" {letter.lower()}", f"\n{letter.lower()}"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)

            if len(encoded) == 1:
                token_ids.add(encoded[0])

        letter_token_ids[label] = sorted(token_ids)

    missing = [label for label, token_ids in letter_token_ids.items() if not token_ids]

    if missing:
        raise RuntimeError(
            f"No valid single-token answer IDs found for labels: {missing}"
        )

    print(f"  letter_token_ids: {letter_token_ids}")
    print(f"  model ready on {device}")

    return tokenizer, model, device, letter_token_ids


def call_llm(
    prompt: str,
    seed: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Sample one answer label and return the sampled label plus logprobs.

    The model distribution is computed at the answer step. Logprobs are first
    aggregated over tokenization variants of A/B/C/D with logsumexp, then
    renormalized over the four admissible labels.

    The sampled label is drawn from this four-label distribution. The returned
    raw_text is the canonical option letter corresponding to the sampled label.
    """
    import torch

    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    tokenizer, model, device, letter_token_ids = _load()

    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        logits = model(inputs).logits[0, -1, :]

        if temperature > 0:
            logits = logits / temperature

        log_probs = torch.log_softmax(logits, dim=-1)

        first_token_logprobs = {
            label: _logsumexp(
                [log_probs[token_id].item() for token_id in letter_token_ids[label]]
            )
            for label in LABELS
        }

    first_token_probs = _renormalize(first_token_logprobs)
    argmax_label = max(LABELS, key=lambda label: first_token_probs[label])

    if temperature == 0:
        sampled_label = argmax_label
    else:
        sampling_probs = _apply_top_p(first_token_probs, top_p=top_p)

        labels = list(sampling_probs.keys())
        probs_tensor = torch.tensor(
            [sampling_probs[label] for label in labels],
            dtype=torch.float32,
        )

        sampled_index = torch.multinomial(probs_tensor, num_samples=1).item()
        sampled_label = labels[sampled_index]

    raw_text = LABEL_LETTER[sampled_label]

    return {
        "raw_text": raw_text,
        "sampled_label": sampled_label,
        "argmax_label": argmax_label,
        "first_token_logprobs": first_token_logprobs,
        "first_token_probs": first_token_probs,
    }


if __name__ == "__main__":
    test_prompt = (
        "The sky is blue.\n\n"
        "Is the sky blue?\n\n"
        "Choose exactly one of the following options:\n"
        "A. True\n"
        "B. False\n"
        "C. Unknown\n"
        "D. Paradox\n\n"
        "Your answer must be exactly one letter: A, B, C, or D.\n"
        "Do not write anything else.\n"
        "Answer:"
    )

    for seed in range(5):
        out = call_llm(test_prompt, seed=seed, temperature=0.7, top_p=0.9)

        print(f"seed={seed}")
        print(f"  raw_text      = {out['raw_text']!r}")
        print(f"  sampled_label = {out['sampled_label']}")
        print(f"  argmax_label  = {out['argmax_label']}")
        print(f"  probs         = {out['first_token_probs']}")
        print()